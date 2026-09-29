"""Classical multidimensional scaling (Torgerson, 1952 / Gower, 1966).

Given pairwise Euclidean distances (or a feature matrix from which they are
computed), classical MDS double-centres the squared distance matrix and
embeds the points with the leading eigenvectors of the resulting Gram
matrix. When the distances come from a Euclidean configuration of rank
``k``, the first ``k`` coordinates recover that configuration up to
translation, rotation, and reflection.

``fit_transform`` returns the embedding. Classic metric MDS is
transductive; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ClassicalMDS", "mds_reduce"]


class ClassicalMDS:
    """Metric classical MDS via double-centering and eigendecomposition.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    metric :
        ``"euclidean"`` computes pairwise distances from ``X``.
        ``"precomputed"`` treats ``X`` as a square distance matrix.
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    dist_matrix_ :
        Pairwise distance matrix used for the embedding.
    eigenvalues_ :
        Leading eigenvalues of the double-centred Gram matrix (length
        ``n_components_``).
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit`` (``None`` when
        ``metric="precomputed"``).
    reconstruction_error_ :
        Kruskal stress of the embedding against the fitted distances
        (``sqrt(sum (D - D_hat)^2 / sum D^2)``).
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "euclidean",
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.metric = _validate_metric(metric)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.reconstruction_error_: float | None = None

    def fit(self, X) -> ClassicalMDS:
        """Compute a classical MDS embedding of ``X`` and store it."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional embedding of ``X``."""
        k = int(self.n_components)
        if self.metric == "precomputed":
            distances = _as_distance_matrix(X)
            n_samples = distances.shape[0]
            self.n_features_in_ = None
        else:
            matrix = _as_dense(X)
            n_samples, n_features = matrix.shape
            self.n_features_in_ = n_features
            if n_samples < 2:
                raise ValueError("X must contain at least two samples.")
            distances = _pairwise_euclidean(matrix)

        if k >= n_samples:
            raise ValueError(
                f"n_components={k} must be less than n_samples={n_samples}."
            )

        embedding, eigenvalues = _classical_mds(distances, k)
        emb_dist = _pairwise_euclidean(embedding)
        num = float(np.sum((distances - emb_dist) ** 2))
        den = float(np.sum(distances ** 2))
        stress = float(np.sqrt(num / den)) if den > 0.0 else 0.0

        self.embedding_ = np.ascontiguousarray(embedding)
        self.dist_matrix_ = distances
        self.eigenvalues_ = eigenvalues
        self.n_components_ = k
        self.reconstruction_error_ = stress
        return self.embedding_


def mds_reduce(
    X,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int | None = None,
):
    """Fit classical MDS and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = ClassicalMDS(
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"MDS{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_metric(value: object) -> str:
    if value not in ("euclidean", "precomputed"):
        raise ValueError('metric must be "euclidean" or "precomputed".')
    return str(value)


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


def _as_distance_matrix(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "precomputed metric requires a square distance matrix "
            "of shape (n_samples, n_samples)."
        )
    if matrix.shape[0] < 2:
        raise ValueError("X must contain at least two samples.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Distance matrix must contain only finite values.")
    if np.any(matrix < 0.0):
        raise ValueError("Distance matrix must be non-negative.")
    # Symmetrise mild numerical asymmetry.
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def _classical_mds(distances: np.ndarray, n_components: int):
    """Classical MDS (Torgerson scaling) on a distance matrix.

    Returns ``(embedding, eigenvalues)`` where ``eigenvalues`` are the
    leading ``n_components`` eigenvalues of the double-centred Gram matrix.
    """
    n = distances.shape[0]
    d2 = distances * distances
    h = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * h @ d2 @ h
    b = 0.5 * (b + b.T)
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    positives = np.maximum(eigenvalues[:n_components], 0.0)
    embedding = eigenvectors[:, :n_components] * np.sqrt(positives)
    # Deterministic sign: flip each column so its first nonzero entry is >= 0.
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-12)
        if len(nz) and col[nz[0]] < 0.0:
            embedding[:, j] = -col
    return embedding, positives
