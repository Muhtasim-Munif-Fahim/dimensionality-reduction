"""Robust PCA by Principal Component Pursuit (low-rank + sparse decomposition).

Classical PCA is a least-squares fit, so a handful of grossly corrupted
entries (sensor glitches, occlusions, data-entry errors) can rotate every
principal axis.  Principal Component Pursuit (Candès, Li, Ma & Wright 2011)
instead splits the data matrix as ``M = L + S`` with ``L`` low rank and ``S``
sparse by solving the convex program

    min ||L||_* + lam * ||S||_1   subject to   P_obs(L + S) = P_obs(M),

where ``||.||_*`` is the nuclear norm and ``P_obs`` keeps the observed
entries, so missing values (``NaN``) are handled as well (robust matrix
completion).  With ``lam = 1 / sqrt(max(n, d))`` the decomposition is exact
with high probability when ``L`` is incoherent and the corruptions are
sparse and spread out.

The program is solved by the inexact augmented Lagrange multiplier method
(Lin, Chen & Ma 2010): alternating singular-value thresholding for ``L``,
entry-wise soft thresholding for ``S``, and a dual ascent step with an
increasing penalty ``mu``.

``RobustPCA`` wraps this in the usual ``fit`` / ``transform`` API: the
principal axes are those of the recovered low-rank part, ``sparse_`` holds
the detected corruptions and ``outlier_scores_`` their per-row size.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "RobustPCA",
    "principal_component_pursuit",
    "rpca_reduce",
]


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


def _soft_threshold(A: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)


def _svt(A: np.ndarray, tau: float):
    """Singular-value thresholding: the prox of ``tau * ||.||_*``."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    rank = int(np.count_nonzero(s))
    return (U[:, :rank] * s[:rank]) @ Vt[:rank], rank


def principal_component_pursuit(
    M,
    lam: float | None = None,
    mu: float | None = None,
    rho: float = 1.5,
    tol: float = 1e-7,
    max_iter: int = 1000,
):
    """Decompose ``M = L + S`` (low rank + sparse) by inexact ALM.

    ``NaN`` entries of ``M`` are treated as unobserved: the constraint only
    binds on observed entries, ``L`` fills in the missing values and ``S`` is
    zero there.

    Returns ``(L, S, info)`` where ``info`` has ``n_iter``, ``converged``,
    ``rank`` (of ``L``), ``lam``, ``mu`` (initial penalty) and
    ``residual`` (relative Frobenius residual on observed entries).
    """
    D = _as_matrix(M)
    n, d = D.shape
    if n < 2 or d < 2:
        raise ValueError("principal component pursuit needs at least a 2 x 2 matrix.")
    observed = ~np.isnan(D)
    if not observed.any():
        raise ValueError("M has no observed entries.")
    D = np.where(observed, D, 0.0)
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if rho <= 1.0:
        raise ValueError("rho must be greater than 1.")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter at least 1.")

    norm_fro = float(np.linalg.norm(D))
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    info = {"n_iter": 0, "converged": True, "rank": 0, "lam": float(lam), "mu": 0.0, "residual": 0.0}
    if norm_fro == 0.0:
        return L, S, info

    spectral = float(np.linalg.norm(D, 2))
    # Dual initialisation of Lin et al.: Y = D / J(D), J = max(||D||_2, ||D||_inf / lam).
    Y = D / max(spectral, float(np.max(np.abs(D))) / lam)
    if mu is None:
        mu = 1.25 / spectral
    if mu <= 0:
        raise ValueError("mu must be positive.")
    mu0 = float(mu)
    mu_max = mu * 1e7
    rank = 0
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        L, rank = _svt(D - S + Y / mu, 1.0 / mu)
        T = D - L + Y / mu
        S = np.where(observed, _soft_threshold(T, lam / mu), T)
        Z = np.where(observed, D - L - S, 0.0)
        Y = Y + mu * Z
        mu = min(mu * rho, mu_max)
        residual = float(np.linalg.norm(Z)) / norm_fro
        if residual < tol:
            converged = True
            break
    S = np.where(observed, S, 0.0)
    info.update(
        n_iter=it,
        converged=converged,
        rank=int(rank),
        mu=mu0,
        residual=float(np.linalg.norm(np.where(observed, D - L - S, 0.0))) / norm_fro,
    )
    return L, S, info


class RobustPCA:
    """PCA on the low-rank part recovered by Principal Component Pursuit.

    Parameters
    ----------
    n_components :
        Number of principal axes kept for ``transform``.  ``None`` keeps the
        numerical rank of the recovered low-rank matrix.
    lam :
        Sparsity weight; default ``1 / sqrt(max(n, d))``.
    center :
        ``"mean"`` (default) or ``None``.  The decomposition is run on the raw
        matrix (a constant offset only adds one to the rank); the principal
        axes are then those of ``L`` centred on its own column means, which
        are a corruption-free estimate of the data centre.
    tol, max_iter, rho, mu :
        Inexact-ALM settings (see :func:`principal_component_pursuit`).

    Attributes
    ----------
    low_rank_, sparse_ :
        ``L`` and ``S`` with ``X = L + S`` on observed entries (``sparse_``
        is 0 where ``X`` is missing; ``low_rank_`` imputes those entries).
    components_ :
        Orthonormal principal axes of ``L``, shape ``(n_components_, n_features)``.
    singular_values_, explained_variance_, explained_variance_ratio_ :
        Spectrum of ``L`` restricted to the kept axes (ratio is relative to
        the total variance of ``L``).
    center_ :
        Column means of ``low_rank_`` (zeros if ``center=None``).
    rank_ :
        Numerical rank of ``L``.
    outlier_scores_ :
        Euclidean norm of each row of ``sparse_`` (0 for clean rows).
    corruption_fraction_ :
        Fraction of observed entries with a non-zero sparse component.
    n_iter_, converged_ :
        Solver diagnostics.
    """

    def __init__(
        self,
        n_components: int | None = None,
        lam: float | None = None,
        center: str | None = "mean",
        tol: float = 1e-7,
        max_iter: int = 1000,
        rho: float = 1.5,
        mu: float | None = None,
    ) -> None:
        self.n_components = n_components
        self.lam = lam
        self.center = center
        self.tol = tol
        self.max_iter = max_iter
        self.rho = rho
        self.mu = mu
        self.low_rank_ = None
        self.sparse_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.center_ = None
        self.rank_ = None
        self.n_components_ = None
        self.outlier_scores_ = None
        self.corruption_fraction_ = None
        self.n_iter_ = 0
        self.converged_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None) -> "RobustPCA":
        matrix = _as_matrix(X)
        n, d = matrix.shape
        if self.center not in ("mean", None):
            raise ValueError("center must be 'mean' or None.")
        if self.n_components is not None:
            if not isinstance(self.n_components, (int, np.integer)) or not 1 <= self.n_components <= min(n, d):
                raise ValueError("n_components must be an integer in [1, min(n_samples, n_features)].")
        observed = ~np.isnan(matrix)
        if np.any(observed.sum(axis=0) == 0):
            raise ValueError("Every feature needs at least one observed value.")
        L, S, info = principal_component_pursuit(
            matrix,
            lam=self.lam,
            mu=self.mu,
            rho=self.rho,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        center = L.mean(axis=0) if self.center == "mean" else np.zeros(d)
        U, s, Vt = np.linalg.svd(L - center, full_matrices=False)
        cutoff = max(n, d) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        # PCP drives the discarded singular values to (numerically) zero; use a
        # tolerance tied to the solver accuracy rather than machine epsilon.
        cutoff = max(cutoff, 10.0 * self.tol * (s[0] if s.size else 0.0))
        rank = int(np.sum(s > cutoff))
        rank = max(rank, 1)
        q = rank if self.n_components is None else int(self.n_components)
        signs = np.sign(Vt[:q, np.argmax(np.abs(Vt[:q]), axis=1)].diagonal())
        signs[signs == 0] = 1.0
        self.components_ = Vt[:q] * signs[:, None]
        self.singular_values_ = s[:q]
        total = float(np.sum(s**2)) / max(n - 1, 1)
        self.explained_variance_ = s[:q] ** 2 / max(n - 1, 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(q)
        )
        self.low_rank_ = L
        self.sparse_ = S
        self.center_ = center
        self.rank_ = rank
        self.n_components_ = q
        self.outlier_scores_ = np.linalg.norm(S, axis=1)
        nnz = np.count_nonzero(np.abs(S[observed]) > 1e-9 * max(1.0, float(np.max(np.abs(matrix[observed])))))
        self.corruption_fraction_ = nnz / float(observed.sum())
        self.n_iter_ = info["n_iter"]
        self.converged_ = info["converged"]
        self.n_features_in_ = d
        return self

    def _check_is_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("This RobustPCA instance is not fitted yet.")

    def transform(self, X) -> np.ndarray:
        """Project complete rows onto the robust principal axes."""
        self._check_is_fitted()
        matrix = _as_matrix(X)
        if np.isnan(matrix).any():
            raise ValueError("transform needs complete rows; use fit_transform for missing data.")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        return (matrix - self.center_) @ self.components_.T

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit, then return the scores of the *cleaned* training rows ``L``."""
        self.fit(X)
        return (self.low_rank_ - self.center_) @ self.components_.T

    def inverse_transform(self, Z) -> np.ndarray:
        self._check_is_fitted()
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.center_

    def clean(self) -> np.ndarray:
        """Training data with corruptions removed and missing entries imputed."""
        self._check_is_fitted()
        return self.low_rank_.copy()


def rpca_reduce(X, n_components: int | None = 2, **kwargs):
    """Fit RobustPCA and return the cleaned-row scores as a DataFrame."""
    import pandas as pd

    model = RobustPCA(n_components=n_components, **kwargs)
    emb = model.fit_transform(X)
    columns = [f"RPCA{i + 1}" for i in range(model.n_components_)]
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(emb, index=index, columns=columns)
