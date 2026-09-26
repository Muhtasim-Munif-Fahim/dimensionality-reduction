"""Isomap manifold embedding in NumPy (Tenenbaum, de Silva & Langford, 2000).

PCA / TruncatedSVD / FastICA / NMF are linear or parts-based factorizations.
Isomap instead approximates geodesic distances on a neighbourhood graph and
embeds them with classical multidimensional scaling (MDS). This module is
sklearn-free: k-nearest neighbours, Dijkstra geodesics, and the MDS
eigendecomposition are all implemented with NumPy.

``fit_transform`` returns the embedding. Classic Isomap is a transductive
map; there is no out-of-sample ``transform``. When the raw kNN graph is
disconnected, shortest inter-component Euclidean edges are added so the
geodesic matrix stays finite.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Isomap", "isomap_reduce"]


class Isomap:
    """Isometric Mapping via a geodesic kNN graph and classical MDS.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    n_neighbors :
        Number of nearest neighbours used to build the neighbourhood graph.
        Must be at least 1 and strictly less than ``n_samples``.
    metric :
        Distance used for the kNN graph. Only ``"euclidean"`` is supported.
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    dist_matrix_ :
        Geodesic distance matrix used for MDS.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    reconstruction_error_ :
        Kruskal stress of the classical MDS embedding
        (``sqrt(sum (D - D_hat)^2 / sum D^2)`` on pairwise distances).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.metric = _validate_metric(metric)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.reconstruction_error_: float | None = None

    def fit(self, X) -> Isomap:
        """Compute an Isomap embedding of ``X`` and store it on ``embedding_``."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional embedding of ``X``."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")
        k = int(self.n_components)
        if k >= n_samples:
            raise ValueError(
                f"n_components={k} must be less than n_samples={n_samples}."
            )
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be less than "
                f"n_samples={n_samples}."
            )

        euclidean = _pairwise_euclidean(matrix)
        graph = _knn_graph(euclidean, self.n_neighbors)
        geodesic = _floyd_warshall(graph)
        if not np.all(np.isfinite(geodesic)):
            raise ValueError(
                "The neighbourhood graph is disconnected; increase "
                "n_neighbors or check that the data form one manifold."
            )

        embedding = _classical_mds(geodesic, k)
        # Kruskal stress against the geodesic distances.
        emb_dist = _pairwise_euclidean(embedding)
        num = float(np.sum((geodesic - emb_dist) ** 2))
        den = float(np.sum(geodesic ** 2))
        stress = float(np.sqrt(num / den)) if den > 0.0 else 0.0

        self.embedding_ = np.ascontiguousarray(embedding)
        self.dist_matrix_ = geodesic
        self.n_components_ = k
        self.n_features_in_ = n_features
        self.reconstruction_error_ = stress
        return self.embedding_


def isomap_reduce(
    X,
    n_components: int = 2,
    n_neighbors: int = 5,
    random_state: int | None = None,
):
    """Fit Isomap and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"ISO{i + 1}" for i in range(model.n_components_)]
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


def _validate_metric(value: object) -> str:
    if value != "euclidean":
        raise ValueError('metric must be "euclidean"')
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


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    # Squared distances via (x-y)^2 = ||x||^2 + ||y||^2 - 2 x·y, then sqrt.
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def _knn_graph(distances: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Symmetrised kNN graph with Euclidean edge weights; others are inf.

    After the mutual kNN edges are placed, any remaining connected-component
    gaps are closed with the shortest inter-component Euclidean edges (a
    forest-of-MST bridge). That keeps Floyd–Warshall defined on one manifold
    even when the raw kNN graph splits well-separated clusters.
    """
    n = distances.shape[0]
    graph = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    # Exclude self: argsort each row and take the next n_neighbors.
    order = np.argsort(distances, axis=1)
    for i in range(n):
        for j in order[i, 1 : n_neighbors + 1]:
            graph[i, j] = distances[i, j]
            graph[j, i] = distances[i, j]
    return _bridge_components(graph, distances)


def _bridge_components(graph: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Add shortest inter-component edges until the graph is connected."""
    n = graph.shape[0]

    def labels() -> np.ndarray:
        lab = -np.ones(n, dtype=int)
        current = 0
        for start in range(n):
            if lab[start] >= 0:
                continue
            stack = [start]
            lab[start] = current
            while stack:
                node = stack.pop()
                for nbr in range(n):
                    if lab[nbr] < 0 and np.isfinite(graph[node, nbr]) and node != nbr:
                        lab[nbr] = current
                        stack.append(nbr)
            current += 1
        return lab

    while True:
        lab = labels()
        n_comp = int(lab.max()) + 1
        if n_comp <= 1:
            return graph
        # Shortest edge between every pair of components; add the global min
        # among those bridges, then repeat (Kruskal-style).
        best = None  # (dist, i, j)
        for i in range(n):
            for j in range(i + 1, n):
                if lab[i] == lab[j]:
                    continue
                d = float(distances[i, j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            return graph
        _, i, j = best
        graph[i, j] = distances[i, j]
        graph[j, i] = distances[i, j]


def _floyd_warshall(graph: np.ndarray) -> np.ndarray:
    dist = graph.copy()
    n = dist.shape[0]
    for k in range(n):
        # dist = min(dist, dist[:, k:k+1] + dist[k:k+1, :])
        via = dist[:, k : k + 1] + dist[k : k + 1, :]
        np.minimum(dist, via, out=dist)
    return dist


def _classical_mds(distances: np.ndarray, n_components: int) -> np.ndarray:
    """Classical MDS (Torgerson scaling) on a squared distance matrix."""
    n = distances.shape[0]
    d2 = distances * distances
    # Double centering: B = -1/2 H D2 H
    h = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * h @ d2 @ h
    b = 0.5 * (b + b.T)
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Keep the leading positive eigenvalues.
    positives = np.maximum(eigenvalues[:n_components], 0.0)
    return eigenvectors[:, :n_components] * np.sqrt(positives)
