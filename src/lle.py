"""Locally Linear Embedding in NumPy (Roweis & Saul, 2000).

PCA / TruncatedSVD / FastICA / NMF are linear or parts-based factorizations;
Isomap preserves geodesic distances. LLE instead reconstructs each point as a
linear combination of its neighbours and finds a low-dimensional embedding
that preserves those reconstruction weights. This module is sklearn-free:
k-nearest neighbours, the constrained least-squares weight solve, and the
sparse eigenproblem on ``M = (I - W)ᵀ(I - W)`` are all implemented with NumPy.

``fit_transform`` returns the embedding. Classic LLE is a transductive map;
there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LLE", "lle_reduce"]


class LLE:
    """Locally Linear Embedding via neighbourhood reconstruction weights.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    n_neighbors :
        Number of nearest neighbours used to reconstruct each point.
        Must be at least 1 and strictly less than ``n_samples``.
    reg :
        Regularisation added to the local Gram diagonal when it is
        rank-deficient (``reg * trace(G)`` fallback when the trace is 0).
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    reconstruction_weights_ :
        Sparse dense weight matrix ``W`` of shape ``(n_samples, n_samples)``.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    reconstruction_error_ :
        Mean squared reconstruction error of ``X`` under ``W``
        (``mean_i ||x_i - sum_j W_ij x_j||^2``).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        reg: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.reg = _validate_reg(reg)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.reconstruction_weights_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.reconstruction_error_: float | None = None

    def fit(self, X) -> LLE:
        """Compute an LLE embedding of ``X`` and store it on ``embedding_``."""
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

        knn = _knn_indices(matrix, self.n_neighbors)
        weights = _reconstruction_weights(matrix, knn, self.reg)
        embedding = _embed_from_weights(weights, k_comp)

        # Mean reconstruction error in the original space.
        recon = weights @ matrix
        err = float(np.mean(np.sum((matrix - recon) ** 2, axis=1)))

        self.embedding_ = np.ascontiguousarray(embedding)
        self.reconstruction_weights_ = weights
        self.n_components_ = k_comp
        self.n_features_in_ = n_features
        self.reconstruction_error_ = err
        return self.embedding_


def lle_reduce(
    X,
    n_components: int = 2,
    n_neighbors: int = 5,
    reg: float = 1e-3,
    random_state: int | None = None,
):
    """Fit LLE and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = LLE(
        n_components=n_components,
        n_neighbors=n_neighbors,
        reg=reg,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"LLE{i + 1}" for i in range(model.n_components_)]
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


def _validate_reg(value: object) -> float:
    try:
        reg = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("reg must be a non-negative float.") from exc
    if not np.isfinite(reg) or reg < 0.0:
        raise ValueError("reg must be a non-negative float.")
    return reg


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


def _pairwise_squared_euclidean(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _knn_indices(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    d2 = _pairwise_squared_euclidean(X)
    # argsort each row; skip self at position 0.
    order = np.argsort(d2, axis=1)
    return order[:, 1 : n_neighbors + 1]


def _reconstruction_weights(
    X: np.ndarray, knn: np.ndarray, reg: float
) -> np.ndarray:
    """Solve sum-to-one constrained least squares for each point's neighbours."""
    n, _ = X.shape
    k = knn.shape[1]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = knn[i]
        Z = X[nbrs] - X[i]  # (k, p)
        G = Z @ Z.T  # (k, k)
        # Regularise so the local Gram is invertible when k > p or points
        # are nearly coplanar.
        trace = float(np.trace(G))
        if trace > 0.0:
            G = G + reg * trace * np.eye(k)
        else:
            G = G + reg * np.eye(k)
        ones = np.ones(k)
        try:
            w = np.linalg.solve(G, ones)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G, ones, rcond=None)[0]
        w_sum = float(w.sum())
        if abs(w_sum) < 1e-12:
            w = np.full(k, 1.0 / k)
        else:
            w = w / w_sum
        W[i, nbrs] = w
    return W


def _embed_from_weights(W: np.ndarray, n_components: int) -> np.ndarray:
    """Bottom eigenvectors of M = (I - W)ᵀ(I - W), skipping the constant mode."""
    n = W.shape[0]
    I_W = np.eye(n) - W
    M = I_W.T @ I_W
    M = 0.5 * (M + M.T)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Smallest eigenvalue ≈ 0 (constant embedding); take the next n_components.
    idx = np.argsort(eigenvalues)
    # Guard against numerical noise picking a near-duplicate of the constant.
    chosen = idx[1 : n_components + 1]
    embedding = eigenvectors[:, chosen]
    # Deterministic sign: flip each column so its first nonzero entry is >= 0.
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-12)
        if len(nz) and col[nz[0]] < 0.0:
            embedding[:, j] = -col
    return embedding
