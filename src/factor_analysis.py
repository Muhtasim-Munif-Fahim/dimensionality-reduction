"""Factor Analysis in NumPy (EM / MLE-style).

PCA / TruncatedSVD rotate the leading sample covariance eigenvectors.
Factor Analysis instead models ``X ≈ mean + Z @ components.T + ε`` where
``Z`` are unit-variance latent factors and ``ε`` has diagonal noise
(``noise_variance_`` / uniquenesses). The EM updates below follow the
classic Rubin & Thayer / sklearn ``FactorAnalysis`` style: E-step
posterior moments of ``Z``, M-step closed-form loadings and uniquenesses.

``fit`` / ``fit_transform`` / ``transform`` return the posterior factor
means. ``factor_analysis_reduce`` mirrors ``mds_reduce`` / ``lle_reduce``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FactorAnalysis", "factor_analysis_reduce"]


class FactorAnalysis:
    """Probabilistic Factor Analysis via EM.

    Parameters
    ----------
    n_components :
        Number of latent factors (rank of the common-factor covariance).
    max_iter :
        Maximum EM iterations.
    tol :
        Relative change in log-likelihood stopping tolerance.
    noise_variance_init :
        Optional length-``n_features`` uniqueness start. ``None`` uses the
        sample variance of each column.
    random_state :
        Seed for the SVD-based loading initialisation (tie-breaking via
        sign of the first non-zero entry is deterministic; the seed is
        kept for API symmetry with NMF / t-SNE).

    Attributes
    ----------
    components_ :
        Factor loadings, shape ``(n_components, n_features)``.
    noise_variance_ :
        Per-feature uniquenesses, shape ``(n_features,)``.
    mean_ :
        Per-feature mean subtracted during ``fit``.
    n_iter_ :
        Number of EM iterations run.
    loglike_ :
        Log-likelihood after the last EM step (or ``None`` if not fitted).
    n_components_ :
        Number of factors used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-2,
        noise_variance_init=None,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.max_iter = _validate_max_iter(max_iter)
        self.tol = _validate_tol(tol)
        self.noise_variance_init = noise_variance_init
        self.random_state = _validate_random_state(random_state)
        self.components_: np.ndarray | None = None
        self.noise_variance_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_iter_: int | None = None
        self.loglike_: float | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> FactorAnalysis:
        """Estimate loadings and uniquenesses from ``X``."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        n_components = int(self.n_components)
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")
        if n_components > n_features:
            raise ValueError(
                f"n_components={n_components} cannot exceed "
                f"n_features={n_features}."
            )

        mean = matrix.mean(axis=0)
        Xc = matrix - mean
        # Sample covariance (ML, divide by n).
        cov = (Xc.T @ Xc) / float(n_samples)

        # Initialise uniquenesses.
        if self.noise_variance_init is None:
            psi = np.maximum(np.diag(cov).copy(), 1e-10)
        else:
            psi = np.asarray(self.noise_variance_init, dtype=float).ravel()
            if psi.shape != (n_features,):
                raise ValueError(
                    "noise_variance_init must have shape (n_features,)."
                )
            if not np.all(np.isfinite(psi)) or np.any(psi <= 0.0):
                raise ValueError("noise_variance_init must be positive and finite.")

        # Initialise loadings from the leading eigenvectors of cov (scaled).
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order[:n_components]], 0.0)
        evecs = evecs[:, order[:n_components]]
        # W ≈ U * sqrt(max(λ - ψ_mean, 0)) in the classic SVD start.
        psi_mean = float(np.mean(psi))
        scale = np.sqrt(np.maximum(evals - psi_mean, 0.0))
        W = evecs * scale  # (n_features, n_components)
        # Orient columns for determinism.
        for j in range(n_components):
            idx = int(np.argmax(np.abs(W[:, j])))
            if W[idx, j] < 0.0:
                W[:, j] *= -1.0

        prev_ll = -np.inf
        ll = -np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # E-step: posterior cov of Z is (I + W.T Ψ^{-1} W)^{-1}
            psi_safe = np.maximum(psi, 1e-12)
            inv_psi = 1.0 / psi_safe
            Wt_invPsi = W.T * inv_psi  # (k, p)
            G = np.eye(n_components) + Wt_invPsi @ W  # (k, k)
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                G_inv = np.linalg.pinv(G)
            # E[Z|X] for all rows: Xc @ invPsi @ W @ G_inv  -> (n, k)
            # beta = G_inv @ W.T @ diag(inv_psi)  -> (k, p)
            beta = G_inv @ Wt_invPsi  # (k, p)
            Ez = Xc @ beta.T  # (n, k)
            # E[ZZ.T] = n * G_inv + Ez.T @ Ez  (sum over samples)
            Ezz = n_samples * G_inv + Ez.T @ Ez

            # M-step: W_new = (Xc.T @ Ez) @ inv(Ezz)
            try:
                Ezz_inv = np.linalg.inv(Ezz)
            except np.linalg.LinAlgError:
                Ezz_inv = np.linalg.pinv(Ezz)
            W = (Xc.T @ Ez) @ Ezz_inv  # (p, k)
            # Uniquenesses: diag(cov - W @ (Ez.T @ Xc)/n ) but use
            # psi_i = cov_ii - (W @ (Ezz / n) @ W.T)_ii
            WW = W @ (Ezz / float(n_samples)) @ W.T
            psi = np.maximum(np.diag(cov) - np.diag(WW), 1e-10)

            # Log-likelihood under FA model: -0.5 * (n*p*log(2π) + n*log|Σ|
            # + n * tr(Σ^{-1} cov)) with Σ = W W.T + diag(ψ)
            ll = _fa_loglike(cov, W, psi, n_samples)
            if prev_ll > -np.inf:
                diff = ll - prev_ll
                # Relative improvement; stop when tiny (or numerical decrease).
                scale_ll = max(abs(prev_ll), 1.0)
                if abs(diff) / scale_ll < self.tol:
                    prev_ll = ll
                    break
            prev_ll = ll

        self.mean_ = mean
        self.components_ = W.T.copy()  # (n_components, n_features)
        self.noise_variance_ = psi.copy()
        self.n_iter_ = n_iter
        self.loglike_ = float(ll)
        self.n_components_ = n_components
        self.n_features_in_ = n_features
        return self

    def transform(self, X) -> np.ndarray:
        """Return posterior factor means for ``X``."""
        self._check_is_fitted()
        matrix = _as_dense(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but FactorAnalysis was "
                f"fitted on {self.n_features_in_}."
            )
        Xc = matrix - self.mean_
        W = self.components_.T  # (p, k)
        psi = np.maximum(self.noise_variance_, 1e-12)
        inv_psi = 1.0 / psi
        Wt_invPsi = W.T * inv_psi
        G = np.eye(self.n_components_) + Wt_invPsi @ W
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        beta = G_inv @ Wt_invPsi
        return Xc @ beta.T

    def fit_transform(self, X) -> np.ndarray:
        """Fit the model and return the factor scores for ``X``."""
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("This FactorAnalysis instance is not fitted yet.")


def factor_analysis_reduce(
    X,
    n_components: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-2,
    random_state: int | None = None,
):
    """Fit FactorAnalysis and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = FactorAnalysis(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"FA{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_max_iter(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("max_iter must be a positive integer.")
    return int(value)


def _validate_tol(value: object) -> float:
    try:
        tol = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("tol must be a non-negative float.") from exc
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a non-negative float.")
    return tol


def _validate_random_state(value: object) -> int | None:
    if value is None:
        return None
    if not _is_integral(value):
        raise ValueError("random_state must be an int or None.")
    return int(value)


def _as_dense(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")
    return matrix


def _fa_loglike(
    cov: np.ndarray, W: np.ndarray, psi: np.ndarray, n_samples: int
) -> float:
    """Gaussian FA log-likelihood (up to additive constants that cancel in Δ)."""
    n_features = cov.shape[0]
    psi_safe = np.maximum(psi, 1e-12)
    # Σ = W W.T + diag(ψ); use matrix-determinant lemma / Woodbury for speed
    # but for kit sizes a dense path is fine.
    Sigma = W @ W.T + np.diag(psi_safe)
    try:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -np.inf
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf
    # ll = -0.5 * (n * logdet + n * tr(Σ^{-1} cov) + n * p * log(2π))
    # Drop the constant n*p*log(2π) for monitoring.
    trace_term = float(np.trace(Sigma_inv @ cov))
    return float(-0.5 * n_samples * (logdet + trace_term))
