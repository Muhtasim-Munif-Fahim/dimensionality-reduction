"""Spectral Embedding / Laplacian Eigenmaps in NumPy (Belkin & Niyogi, 2003).

PCA / TruncatedSVD / FastICA / NMF are linear or parts-based factorizations;
Isomap preserves geodesic distances; LLE preserves reconstruction weights.
Spectral Embedding instead builds a k-nearest-neighbour affinity graph, forms
the (symmetric normalised) graph Laplacian, and embeds points with the
smallest non-trivial eigenvectors. This module is sklearn-free: kNN affinity,
degree-normalised Laplacian, and the dense eigendecomposition are all
implemented with NumPy.

``fit_transform`` returns the embedding. Classic Laplacian Eigenmaps is a
transductive map; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpectralEmbedding", "spectral_embedding", "spectral_reduce"]


class SpectralEmbedding:
    """Laplacian Eigenmaps via a kNN affinity graph.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    n_neighbors :
        Number of nearest neighbours used to build the affinity graph.
        Must be at least 1 and strictly less than ``n_samples``.
    affinity :
        ``"nearest_neighbors"`` (binary / unweighted) or ``"rbf"`` (Gaussian
        heat kernel on the same kNN edges).
    gamma :
        RBF bandwidth for ``affinity="rbf"``. ``None`` uses
        ``1 / median(kNN distances)^2`` (or 1 when that median is 0).
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    affinity_matrix_ :
        Symmetric affinity / weight matrix ``W``.
    eigenvalues_ :
        Smallest non-trivial Laplacian eigenvalues used for the embedding.
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
        gamma: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.affinity = _validate_affinity(affinity)
        self.gamma = _validate_gamma(gamma)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.affinity_matrix_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> SpectralEmbedding:
        """Compute a spectral embedding of ``X`` and store it on ``embedding_``."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional embedding of ``X``."""
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
        # Need room for the constant eigenvector we discard.
        if k_comp + 1 >= n_samples:
            raise ValueError(
                f"n_components={k_comp} leaves no room for the trivial "
                f"eigenvector (need n_components + 1 < n_samples={n_samples})."
            )

        W = _knn_affinity(matrix, self.n_neighbors, self.affinity, self.gamma)
        embedding, eigenvalues = _laplacian_embedding(W, k_comp)

        self.embedding_ = np.ascontiguousarray(embedding)
        self.affinity_matrix_ = W
        self.eigenvalues_ = eigenvalues
        self.n_components_ = k_comp
        self.n_features_in_ = n_features
        return self.embedding_


def spectral_embedding(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    affinity: str = "nearest_neighbors",
    gamma: float | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Fit SpectralEmbedding and return the ndarray embedding."""
    model = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity=affinity,
        gamma=gamma,
        random_state=random_state,
    )
    return model.fit_transform(X)


def spectral_reduce(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    affinity: str = "nearest_neighbors",
    gamma: float | None = None,
    random_state: int | None = None,
):
    """Fit SpectralEmbedding and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity=affinity,
        gamma=gamma,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"SE{i + 1}" for i in range(model.n_components_)]
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


def _validate_gamma(value: object) -> float | None:
    if value is None:
        return None
    try:
        gamma = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma must be a positive float or None.") from exc
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive float or None.")
    return gamma


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


def _knn_affinity(
    X: np.ndarray,
    n_neighbors: int,
    affinity: str,
    gamma: float | None,
) -> np.ndarray:
    """Symmetric kNN affinity matrix (binary or RBF-weighted)."""
    n = X.shape[0]
    D = _pairwise_euclidean(X)
    # argsort; skip self at position 0
    order = np.argsort(D, axis=1)
    knn = order[:, 1 : n_neighbors + 1]
    knn_dists = np.take_along_axis(D, knn, axis=1)

    if affinity == "rbf":
        if gamma is None:
            med = float(np.median(knn_dists))
            gamma = 1.0 / (med * med) if med > 1e-12 else 1.0
        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j_pos, j in enumerate(knn[i]):
                w = float(np.exp(-gamma * knn_dists[i, j_pos] ** 2))
                W[i, j] = max(W[i, j], w)
                W[j, i] = max(W[j, i], w)
    else:
        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in knn[i]:
                W[i, j] = 1.0
                W[j, i] = 1.0
        np.fill_diagonal(W, 0.0)
    return W


def _laplacian_embedding(W: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest non-trivial eigenvectors of the symmetric normalised Laplacian."""
    n = W.shape[0]
    degrees = W.sum(axis=1)
    # Avoid divide-by-zero for isolated nodes (should be rare with mutual kNN).
    degrees = np.where(degrees > 0.0, degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    # L_sym = I - D^{-1/2} W D^{-1/2}
    DWD = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L = np.eye(n) - DWD
    # Dense eigh; eigenvalues ascending.
    evals, evecs = np.linalg.eigh(L)
    # Drop the trivial (near-zero) eigenvector; take the next n_components.
    # Guard against numerical noise on the constant mode.
    start = 1
    if start + n_components > n:
        raise ValueError("Not enough eigenvectors for the requested n_components.")
    embedding = evecs[:, start : start + n_components]
    eigenvalues = evals[start : start + n_components]
    return embedding, eigenvalues
