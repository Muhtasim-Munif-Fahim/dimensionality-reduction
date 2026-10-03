"""Diffusion Maps in NumPy (Coifman & Lafon, 2006).

Spectral Embedding uses the graph Laplacian; Diffusion Maps instead builds a
Gaussian affinity, optionally density-normalises it (``alpha``), forms the
row-stochastic diffusion / Markov matrix, and embeds points with
``lambda^t * eigenvector`` coordinates (skipping the trivial first mode).
This module is sklearn-free: affinity, Markov normalisation, and the dense
eigendecomposition are all implemented with NumPy.

``fit_transform`` returns the embedding. Classic Diffusion Maps is a
transductive map; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DiffusionMaps", "diffusion_maps", "diffusion_reduce"]


class DiffusionMaps:
    """Diffusion Maps via a Gaussian affinity / Markov matrix.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    n_neighbors :
        Number of nearest neighbours used to build a sparse Gaussian
        affinity. Ignored when ``epsilon`` is set and ``affinity="rbf"``
        builds a dense kernel (still used as a kNN graph when
        ``affinity="nearest_neighbors"``). Must be at least 1 and strictly
        less than ``n_samples``.
    affinity :
        ``"nearest_neighbors"`` (Gaussian weights on kNN edges) or
        ``"rbf"`` (dense Gaussian heat kernel with bandwidth ``epsilon``).
    epsilon :
        Gaussian bandwidth. ``None`` uses the median of the kNN distances
        (or the median of all pairwise distances for dense RBF).
    t :
        Diffusion time. Eigenvalues are raised to this power in the
        embedding (``lambda ** t``).
    alpha :
        Density normalisation exponent (Coifman–Lafon). ``0`` skips
        normalisation; ``0.5`` / ``1`` are common anisotropic choices.
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    affinity_matrix_ :
        Symmetric affinity / weight matrix ``K`` (before Markov row-norm).
    eigenvalues_ :
        Non-trivial diffusion eigenvalues used for the embedding.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        affinity: str = "nearest_neighbors",
        epsilon: float | None = None,
        t: float = 1.0,
        alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.affinity = _validate_affinity(affinity)
        self.epsilon = _validate_epsilon(epsilon)
        self.t = _validate_t(t)
        self.alpha = _validate_alpha(alpha)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.affinity_matrix_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> DiffusionMaps:
        """Compute a diffusion-map embedding of ``X`` and store it on ``embedding_``."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional diffusion-map embedding of ``X``."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")
        k_comp = int(self.n_components)
        if k_comp >= n_samples:
            raise ValueError(
                f"n_components={k_comp} must be less than n_samples={n_samples}."
            )
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be less than "
                f"n_samples={n_samples}."
            )
        if k_comp + 1 >= n_samples:
            raise ValueError(
                f"n_components={k_comp} leaves no room for the trivial "
                f"eigenvector (need n_components + 1 < n_samples={n_samples})."
            )

        K = _gaussian_affinity(
            matrix, self.n_neighbors, self.affinity, self.epsilon
        )
        embedding, eigenvalues = _diffusion_embedding(
            K, k_comp, t=self.t, alpha=self.alpha
        )

        self.embedding_ = np.ascontiguousarray(embedding)
        self.affinity_matrix_ = K
        self.eigenvalues_ = eigenvalues
        self.n_components_ = k_comp
        self.n_features_in_ = n_features
        return self.embedding_


def diffusion_maps(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    affinity: str = "nearest_neighbors",
    epsilon: float | None = None,
    t: float = 1.0,
    alpha: float = 0.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Fit DiffusionMaps and return the ndarray embedding."""
    model = DiffusionMaps(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity=affinity,
        epsilon=epsilon,
        t=t,
        alpha=alpha,
        random_state=random_state,
    )
    return model.fit_transform(X)


def diffusion_reduce(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    affinity: str = "nearest_neighbors",
    epsilon: float | None = None,
    t: float = 1.0,
    alpha: float = 0.0,
    random_state: int | None = None,
):
    """Fit DiffusionMaps and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = DiffusionMaps(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity=affinity,
        epsilon=epsilon,
        t=t,
        alpha=alpha,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"DM{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_n_neighbors(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    return int(value)


def _validate_affinity(value: object) -> str:
    affinity = str(value)
    if affinity not in ("nearest_neighbors", "rbf"):
        raise ValueError("affinity must be 'nearest_neighbors' or 'rbf'.")
    return affinity


def _validate_epsilon(value: object) -> float | None:
    if value is None:
        return None
    try:
        eps = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be a positive float or None.") from exc
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be a positive float or None.")
    return eps


def _validate_t(value: object) -> float:
    try:
        t = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("t must be a non-negative finite float.") from exc
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be a non-negative finite float.")
    return t


def _validate_alpha(value: object) -> float:
    try:
        alpha = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a finite float.") from exc
    if not np.isfinite(alpha):
        raise ValueError("alpha must be a finite float.")
    return alpha


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


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def _gaussian_affinity(
    X: np.ndarray,
    n_neighbors: int,
    affinity: str,
    epsilon: float | None,
) -> np.ndarray:
    """Symmetric Gaussian affinity (kNN-sparse or dense RBF)."""
    n = X.shape[0]
    D = _pairwise_euclidean(X)
    order = np.argsort(D, axis=1)
    knn = order[:, 1 : n_neighbors + 1]
    knn_dists = np.take_along_axis(D, knn, axis=1)

    if affinity == "rbf":
        if epsilon is None:
            # Median pairwise distance (exclude zeros on diagonal).
            off = D[np.triu_indices(n, k=1)]
            med = float(np.median(off)) if off.size else 0.0
            epsilon = med if med > 1e-12 else 1.0
        denom = 2.0 * (epsilon ** 2)
        K = np.exp(-(D ** 2) / denom)
        np.fill_diagonal(K, 0.0)
        return K

    # nearest_neighbors: Gaussian weights on mutual kNN edges
    if epsilon is None:
        med = float(np.median(knn_dists))
        epsilon = med if med > 1e-12 else 1.0
    denom = 2.0 * (epsilon ** 2)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_pos, j in enumerate(knn[i]):
            w = float(np.exp(-(knn_dists[i, j_pos] ** 2) / denom))
            K[i, j] = max(K[i, j], w)
            K[j, i] = max(K[j, i], w)
    np.fill_diagonal(K, 0.0)
    return K


def _diffusion_embedding(
    K: np.ndarray,
    n_components: int,
    t: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-trivial diffusion coordinates from affinity ``K``."""
    n = K.shape[0]
    K = np.array(K, dtype=float, copy=True)
    np.fill_diagonal(K, 0.0)

    # Density / anisotropic normalisation (Coifman–Lafon alpha).
    if abs(alpha) > 1e-15:
        q = K.sum(axis=1)
        q = np.where(q > 0.0, q, 1.0)
        q_alpha = q ** alpha
        K = K / np.outer(q_alpha, q_alpha)

    degrees = K.sum(axis=1)
    degrees = np.where(degrees > 0.0, degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    # Symmetric normalised kernel A = D^{-1/2} K D^{-1/2}; eigenvalues match
    # those of the row-stochastic P = D^{-1} K. Right eigenvectors of P are
    # psi = D^{-1/2} phi.
    A = (d_inv_sqrt[:, None] * K) * d_inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(A)
    # Descending order (largest diffusion eigenvalue first ≈ 1).
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    start = 1  # skip trivial stationary mode
    if start + n_components > n:
        raise ValueError("Not enough eigenvectors for the requested n_components.")
    lam = evals[start : start + n_components]
    phi = evecs[:, start : start + n_components]
    psi = d_inv_sqrt[:, None] * phi
    # Diffusion coordinates: lambda^t * psi
    if t == 0.0:
        scales = np.ones_like(lam)
    else:
        # Clamp tiny negative numerical eigenvalues before powering.
        lam_safe = np.clip(lam, 0.0, None)
        scales = lam_safe ** t
    embedding = psi * scales[None, :]
    return embedding, lam
