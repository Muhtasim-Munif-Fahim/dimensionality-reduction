"""Probabilistic PCA (Tipping & Bishop) with automatic dimensionality.

PPCA is the isotropic-noise special case of Factor Analysis:
``x = mean + W z + eps`` with ``z ~ N(0, I_q)`` and ``eps ~ N(0, sigma2 I_d)``,
so the model covariance is ``C = W W.T + sigma2 I``. Unlike plain PCA it is
a proper density, which gives

* a closed-form maximum-likelihood fit: ``sigma2`` is the mean of the
  discarded covariance eigenvalues and ``W = U_q (L_q - sigma2 I)^(1/2)``;
* per-sample log-likelihoods (``score_samples`` / ``score``) for model
  comparison, novelty scoring and held-out evaluation;
* principled choice of ``q``: Minka's (2000) Laplace approximation to the
  Bayesian evidence (``n_components="mle"``) or BIC (``"bic"``);
* an EM algorithm that fits directly on data with missing values
  (``NaN``), marginalising the unobserved entries exactly, plus
  ``impute`` to fill them with their conditional expectations.

``transform`` returns posterior latent means ``M^-1 W.T (x - mean)`` with
``M = W.T W + sigma2 I``. ``ppca_reduce`` mirrors ``factor_analysis_reduce``.
"""

from __future__ import annotations

from math import lgamma, log, pi

import numpy as np

__all__ = [
    "ProbabilisticPCA",
    "ppca_reduce",
    "minka_log_evidence",
    "minka_mle_dimension",
]


class ProbabilisticPCA:
    """Probabilistic PCA fitted by closed-form MLE or, with NaNs, by EM.

    Parameters
    ----------
    n_components :
        Latent dimension ``q`` (``1 <= q < n_features``), ``"mle"`` for
        Minka's Laplace-evidence choice, or ``"bic"`` for the BIC choice.
    ddof :
        Degrees of freedom for the sample covariance (``0`` is the exact
        MLE; ``1`` reproduces scikit-learn's ``PCA`` scaling).
    max_iter, tol :
        EM settings used only when ``X`` has missing values; ``tol`` is the
        absolute change in mean observed-data log-likelihood per sample.
    random_state :
        Seed for the EM loading initialisation.

    Attributes
    ----------
    components_ :
        Orthonormal principal axes, shape ``(n_components, n_features)``.
    explained_variance_ :
        Retained covariance eigenvalues (closed-form fits).
    loadings_ :
        ``W.T``, shape ``(n_components, n_features)``.
    noise_variance_ :
        Isotropic noise variance ``sigma2``.
    mean_ :
        Feature means.
    n_components_ :
        Latent dimension used.
    evidence_ :
        Per-``q`` selection criterion (log evidence for ``"mle"``,
        ``-BIC/2`` for ``"bic"``), indexed by ``q - 1``; ``None`` otherwise.
    loglike_history_ :
        Mean observed-data log-likelihood after each EM sweep (EM fits).
    n_iter_ :
        EM sweeps run (``0`` for closed-form fits).
    """

    def __init__(
        self,
        n_components: int | str = 2,
        ddof: int = 0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.ddof = ddof
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.loadings_ = None
        self.noise_variance_ = None
        self.mean_ = None
        self.n_components_ = None
        self.evidence_ = None
        self.loglike_history_ = None
        self.n_iter_ = 0
        self.n_features_in_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        matrix = _as_matrix(X)
        n, d = matrix.shape
        if d < 2:
            raise ValueError("ProbabilisticPCA needs at least two features.")
        if self.ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        if n - self.ddof < 2:
            raise ValueError("Need at least two samples (after ddof).")
        missing = np.isnan(matrix)
        if missing.any():
            if isinstance(self.n_components, str):
                raise ValueError(
                    "Automatic n_components needs complete data; pass an integer "
                    "when X has missing values."
                )
            q = _validate_q(self.n_components, d)
            self._fit_em(matrix, ~missing, q)
        else:
            self._fit_closed_form(matrix)
        self.n_features_in_ = d
        return self

    def _fit_closed_form(self, matrix: np.ndarray) -> None:
        n, d = matrix.shape
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        cov = centered.T @ centered / (n - self.ddof)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        eigvecs = eigvecs * _sign_flip(eigvecs)

        self.evidence_ = None
        if isinstance(self.n_components, str) and self.n_components not in ("mle", "bic"):
            raise ValueError("n_components must be an integer, 'mle', or 'bic'.")
        if self.n_components == "mle":
            self.evidence_ = np.array(
                [minka_log_evidence(eigvals, q, n) for q in range(1, d)]
            )
            q = int(np.argmax(self.evidence_)) + 1
        elif self.n_components == "bic":
            self.evidence_ = np.array(
                [_ppca_loglike_from_spectrum(eigvals, q, n) - 0.5 * _n_params(d, q) * log(n)
                 for q in range(1, d)]
            )
            q = int(np.argmax(self.evidence_)) + 1
        else:
            q = _validate_q(self.n_components, d)

        sigma2 = float(eigvals[q:].mean())
        lam = eigvals[:q]
        scale = np.sqrt(np.clip(lam - sigma2, 0.0, None))
        self.mean_ = mean
        self.components_ = eigvecs[:, :q].T
        self.explained_variance_ = lam
        self.noise_variance_ = max(sigma2, 1e-12)
        self.loadings_ = self.components_ * scale[:, None]
        self.n_components_ = q
        self.n_iter_ = 0
        self.loglike_history_ = None

    def _fit_em(self, matrix: np.ndarray, observed: np.ndarray, q: int) -> None:
        n, d = matrix.shape
        if not observed.any(axis=0).all():
            raise ValueError("Every feature needs at least one observed value.")
        if not observed.any(axis=1).all():
            raise ValueError("Every sample needs at least one observed value.")
        max_iter = _validate_positive_int(self.max_iter, "max_iter")
        tol = float(self.tol)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a non-negative float.")

        rng = np.random.default_rng(self.random_state)
        mean = np.nanmean(matrix, axis=0)
        filled = np.where(observed, matrix, mean)
        centered = filled - mean
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        W = (vt[:q].T * (s[:q] / np.sqrt(n))) + 1e-3 * rng.standard_normal((d, q))
        sigma2 = float(np.mean(np.var(filled, axis=0))) * 0.5 + 1e-6

        history = []
        previous = -np.inf
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            # E-step: posterior moments of z_n given the observed entries.
            Ez = np.zeros((n, q))
            Ezz = np.zeros((n, q, q))
            for i in range(n):
                o = observed[i]
                Wo = W[o]
                M = Wo.T @ Wo + sigma2 * np.eye(q)
                Minv = np.linalg.inv(M)
                Ez[i] = Minv @ Wo.T @ (matrix[i, o] - mean[o])
                Ezz[i] = sigma2 * Minv + np.outer(Ez[i], Ez[i])
            # M-step: per-feature joint regression for (W_j, mean_j).
            for j in range(d):
                rows = observed[:, j]
                zr = Ez[rows]
                A = np.zeros((q + 1, q + 1))
                A[:q, :q] = Ezz[rows].sum(axis=0)
                A[:q, q] = A[q, :q] = zr.sum(axis=0)
                A[q, q] = rows.sum()
                xj = matrix[rows, j]
                b = np.concatenate([zr.T @ xj, [xj.sum()]])
                sol = np.linalg.solve(A + 1e-12 * np.eye(q + 1), b)
                W[j] = sol[:q]
                mean[j] = sol[q]
            total = 0.0
            for i in range(n):
                o = observed[i]
                r = matrix[i, o] - mean[o] - W[o] @ Ez[i]
                Wo = W[o]
                total += r @ r + np.trace(Wo.T @ Wo @ (Ezz[i] - np.outer(Ez[i], Ez[i])))
            sigma2 = max(float(total / observed.sum()), 1e-12)

            ll = _observed_loglike(matrix, observed, mean, W, sigma2) / n
            history.append(ll)
            if abs(ll - previous) < tol:
                break
            previous = ll

        # Orthogonalise: W = U diag(s) R  ->  report axes U and W in canonical form.
        u, s, _ = np.linalg.svd(W, full_matrices=False)
        u = u * _sign_flip(u)
        self.mean_ = mean
        self.components_ = u.T
        self.explained_variance_ = s**2 + sigma2
        self.loadings_ = self.components_ * s[:, None]
        self.noise_variance_ = sigma2
        self.n_components_ = q
        self.evidence_ = None
        self.loglike_history_ = np.array(history)
        self.n_iter_ = n_iter

    # ------------------------------------------------------------- inference
    def get_covariance(self) -> np.ndarray:
        self._check_is_fitted()
        W = self.loadings_.T
        return W @ W.T + self.noise_variance_ * np.eye(W.shape[0])

    def get_precision(self) -> np.ndarray:
        """Inverse model covariance via the Woodbury identity."""
        self._check_is_fitted()
        W = self.loadings_.T
        s2 = self.noise_variance_
        M = W.T @ W + s2 * np.eye(W.shape[1])
        return (np.eye(W.shape[0]) - W @ np.linalg.solve(M, W.T)) / s2

    def transform(self, X) -> np.ndarray:
        """Posterior latent means ``E[z | x]`` (complete rows)."""
        self._check_is_fitted()
        matrix = self._check_complete(X)
        W = self.loadings_.T
        M = W.T @ W + self.noise_variance_ * np.eye(W.shape[1])
        return np.linalg.solve(M, W.T @ (matrix - self.mean_).T).T

    def fit_transform(self, X, y=None) -> np.ndarray:
        self.fit(X)
        matrix = _as_matrix(X)
        if np.isnan(matrix).any():
            return self._posterior_means_missing(matrix)
        return self.transform(matrix)

    def inverse_transform(self, Z) -> np.ndarray:
        self._check_is_fitted()
        Z = np.asarray(Z, dtype=float)
        return Z @ self.loadings_ + self.mean_

    def score_samples(self, X) -> np.ndarray:
        """Gaussian log-density of each complete row under ``N(mean, C)``."""
        self._check_is_fitted()
        matrix = self._check_complete(X)
        centered = matrix - self.mean_
        precision = self.get_precision()
        _, logdet = np.linalg.slogdet(self.get_covariance())
        d = matrix.shape[1]
        quad = np.einsum("ij,jk,ik->i", centered, precision, centered)
        return -0.5 * (d * log(2 * pi) + logdet + quad)

    def score(self, X, y=None) -> float:
        return float(self.score_samples(X).mean())

    def impute(self, X) -> np.ndarray:
        """Replace NaNs by ``E[x_missing | x_observed]`` under the fitted model."""
        self._check_is_fitted()
        matrix = _as_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        C = self.get_covariance()
        out = matrix.copy()
        for i in np.flatnonzero(np.isnan(matrix).any(axis=1)):
            m = np.isnan(matrix[i])
            o = ~m
            if not o.any():
                out[i, m] = self.mean_[m]
                continue
            coef = np.linalg.solve(C[np.ix_(o, o)], matrix[i, o] - self.mean_[o])
            out[i, m] = self.mean_[m] + C[np.ix_(m, o)] @ coef
        return out

    def _posterior_means_missing(self, matrix: np.ndarray) -> np.ndarray:
        W = self.loadings_.T
        q = W.shape[1]
        Z = np.zeros((matrix.shape[0], q))
        for i, row in enumerate(matrix):
            o = ~np.isnan(row)
            Wo = W[o]
            M = Wo.T @ Wo + self.noise_variance_ * np.eye(q)
            Z[i] = np.linalg.solve(M, Wo.T @ (row[o] - self.mean_[o]))
        return Z

    def _check_complete(self, X) -> np.ndarray:
        matrix = _as_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        if np.isnan(matrix).any():
            raise ValueError("X contains NaN; use impute() first.")
        return matrix

    def _check_is_fitted(self) -> None:
        if self.loadings_ is None:
            raise RuntimeError("This ProbabilisticPCA instance is not fitted yet.")


def ppca_reduce(X, n_components: int | str = 2, **kwargs):
    """Fit ProbabilisticPCA and return the posterior means as a DataFrame."""
    import pandas as pd

    model = ProbabilisticPCA(n_components=n_components, **kwargs)
    emb = model.fit_transform(X)
    columns = [f"PPCA{i + 1}" for i in range(model.n_components_)]
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(emb, index=index, columns=columns)


def minka_log_evidence(spectrum, rank: int, n_samples: int) -> float:
    """Minka's Laplace approximation to ``log p(D | q = rank)`` for PPCA.

    ``spectrum`` holds the sample covariance eigenvalues in decreasing
    order. Follows Minka, "Automatic choice of dimensionality for PCA"
    (NIPS 2000), eq. 30-31: a uniform (Haar) prior on the principal
    subspace, with the noise eigenvalues pooled into their mean ``v``.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    d = spectrum.size
    if not _is_integral(rank) or not 1 <= rank < d:
        raise ValueError("rank must be an integer in [1, n_features - 1].")
    eps = 1e-15
    if spectrum[rank - 1] < eps:
        return -np.inf
    n = float(n_samples)
    pu = -rank * log(2.0) + sum(
        lgamma((d - i + 1) / 2.0) - log(pi) * (d - i + 1) / 2.0 for i in range(1, rank + 1)
    )
    pl = -np.sum(np.log(spectrum[:rank])) * n / 2.0
    v = max(eps, float(spectrum[rank:].sum()) / (d - rank))
    pv = -log(v) * n * (d - rank) / 2.0
    m = d * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * pi) * (m + rank) / 2.0
    hat = spectrum.copy()
    hat[rank:] = v
    i_idx, j_idx = np.triu_indices(d, k=1)
    keep = i_idx < rank
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    pa = np.sum(
        np.log((spectrum[i_idx] - spectrum[j_idx]) * (1.0 / hat[j_idx] - 1.0 / hat[i_idx]))
        + log(n)
    )
    return float(pu + pl + pv + pp - pa / 2.0 - rank * log(n) / 2.0)


def minka_mle_dimension(X, ddof: int = 1) -> int:
    """Latent dimension maximising :func:`minka_log_evidence` for data ``X``."""
    matrix = _as_matrix(X)
    if np.isnan(matrix).any():
        raise ValueError("X must not contain NaN.")
    n = matrix.shape[0]
    centered = matrix - matrix.mean(axis=0)
    spectrum = np.sort(np.clip(np.linalg.eigvalsh(centered.T @ centered / (n - ddof)), 0, None))[::-1]
    evidence = [minka_log_evidence(spectrum, q, n) for q in range(1, spectrum.size)]
    return int(np.argmax(evidence)) + 1


# ---------------------------------------------------------------- helpers
def _ppca_loglike_from_spectrum(eigvals: np.ndarray, q: int, n: int) -> float:
    """Maximised PPCA log-likelihood (Tipping & Bishop eq. 4 at the MLE)."""
    d = eigvals.size
    sigma2 = max(float(eigvals[q:].mean()), 1e-300)
    logdet = float(np.sum(np.log(np.maximum(eigvals[:q], 1e-300)))) + (d - q) * log(sigma2)
    return -0.5 * n * (d * log(2 * pi) + logdet + d)


def _n_params(d: int, q: int) -> float:
    # mean + W up to rotation + sigma2
    return d + d * q - q * (q - 1) / 2.0 + 1.0


def _observed_loglike(matrix, observed, mean, W, sigma2) -> float:
    total = 0.0
    for i in range(matrix.shape[0]):
        o = observed[i]
        Wo = W[o]
        C = Wo @ Wo.T + sigma2 * np.eye(int(o.sum()))
        r = matrix[i, o] - mean[o]
        _, logdet = np.linalg.slogdet(C)
        total += -0.5 * (o.sum() * log(2 * pi) + logdet + r @ np.linalg.solve(C, r))
    return float(total)


def _sign_flip(vectors: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[idx, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return signs


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_q(value: object, n_features: int) -> int:
    if not _is_integral(value) or not 1 <= int(value) < n_features:
        raise ValueError(
            "n_components must be an integer in [1, n_features - 1], 'mle', or 'bic'."
        )
    return int(value)


def _validate_positive_int(value: object, name: str) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _as_matrix(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if np.isinf(matrix).any():
        raise ValueError("X must not contain infinite values.")
    return matrix
