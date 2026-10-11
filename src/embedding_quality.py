"""Rank-based quality metrics for low-dimensional embeddings.

Silhouette and cluster separation (``evaluation.py``) need labels, and
reconstruction error only applies to linear maps. The metrics here are
**unsupervised** and **method-agnostic**: they compare the neighbourhood
ranks of each point in the original space with those in the embedding.

* ``coranking_matrix``: ``Q[k-1, l-1]`` counts pairs ``(i, j)`` where ``j``
  is the ``k``-th neighbour of ``i`` in the input and the ``l``-th in the
  embedding (Lee & Verleysen, 2009).
* ``trustworthiness`` (Venna & Kaski, 2001) penalises *intrusions*:
  embedding neighbours that were far away in the input.
* ``continuity`` penalises *extrusions*: input neighbours that the
  embedding pushed away.
* ``neighborhood_preservation`` (``Q_NX(K)``): the average fraction of the
  ``K`` nearest input neighbours that stay among the ``K`` nearest
  embedding neighbours.
* ``lcmc``: the local continuity meta-criterion, ``Q_NX(K) - K/(n-1)``
  (Chen & Buja, 2009). It subtracts the overlap a random embedding gets.
* ``rnx_curve`` / ``rnx_auc``: the rescaled ``R_NX(K)``, which is 0 for a
  random embedding and 1 for a perfect one, and its area under a log-K
  axis. That area summarises quality across all scales (Lee, Peluffo-Ordóñez
  & Verleysen, 2015).

Ties in distance are broken by index (stable sort), matching
``sklearn.manifold.trustworthiness``. Distances are Euclidean, and all
computation is in NumPy with ``O(n^2)`` memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "coranking_matrix",
    "trustworthiness",
    "continuity",
    "neighborhood_preservation",
    "lcmc",
    "rnx_curve",
    "rnx_auc",
    "embedding_quality",
]


def _as_array(X, name: str) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _pair(X, Y) -> tuple[np.ndarray, np.ndarray]:
    X = _as_array(X, "X")
    Y = _as_array(Y, "embedding")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and embedding must have the same number of rows")
    if X.shape[0] < 4:
        raise ValueError("at least four samples are required")
    return X, Y


def _rank_matrix(X: np.ndarray) -> np.ndarray:
    """``R[i, j]`` = rank (1-based) of ``j`` among the neighbours of ``i``; ``R[i, i] = 0``."""
    sq = np.sum(X * X, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    np.fill_diagonal(d2, -np.inf)
    order = np.argsort(d2, axis=1, kind="stable")
    n = X.shape[0]
    ranks = np.empty((n, n), dtype=np.int64)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    return ranks


def _check_k(k, n: int, name: str = "n_neighbors") -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(k)


def coranking_matrix(X, embedding) -> np.ndarray:
    """Co-ranking matrix of shape ``(n-1, n-1)``; rows are input ranks."""
    X, Y = _pair(X, embedding)
    n = X.shape[0]
    rx = _rank_matrix(X)
    ry = _rank_matrix(Y)
    off = ~np.eye(n, dtype=bool)
    Q = np.zeros((n - 1, n - 1), dtype=np.int64)
    np.add.at(Q, (rx[off] - 1, ry[off] - 1), 1)
    return Q


def _rank_penalty(r_ref: np.ndarray, r_other: np.ndarray, k: int) -> float:
    n = r_ref.shape[0]
    if k >= n / 2.0:
        raise ValueError("n_neighbors must be less than n_samples / 2")
    in_other = (r_other >= 1) & (r_other <= k)
    in_ref = (r_ref >= 1) & (r_ref <= k)
    missing = in_other & ~in_ref
    penalty = float(np.sum(r_ref[missing] - k))
    return 1.0 - 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)) * penalty


def trustworthiness(X, embedding, n_neighbors: int = 5) -> float:
    """Venna-Kaski trustworthiness in ``[0, 1]`` (higher is better).

    Requires ``n_neighbors < n_samples / 2``. Matches
    ``sklearn.manifold.trustworthiness`` with the Euclidean metric.
    """
    X, Y = _pair(X, embedding)
    k = _check_k(n_neighbors, X.shape[0])
    return _rank_penalty(_rank_matrix(X), _rank_matrix(Y), k)


def continuity(X, embedding, n_neighbors: int = 5) -> float:
    """Continuity in ``[0, 1]``: trustworthiness with the two spaces swapped."""
    X, Y = _pair(X, embedding)
    k = _check_k(n_neighbors, X.shape[0])
    return _rank_penalty(_rank_matrix(Y), _rank_matrix(X), k)


def _qnx_all(Q: np.ndarray) -> np.ndarray:
    """``Q_NX(K)`` for ``K = 1 .. n-1`` from a co-ranking matrix."""
    n = Q.shape[0] + 1
    # Sum of the upper-left KxK block for every K via 2-D cumulative sums.
    cum = np.cumsum(np.cumsum(Q, axis=0), axis=1)
    Ks = np.arange(1, n)
    return np.diagonal(cum).astype(float) / (Ks * n)


def neighborhood_preservation(X, embedding, n_neighbors: int = 10) -> float:
    """``Q_NX(K)``: mean overlap fraction of the K-nearest-neighbour sets."""
    Q = coranking_matrix(X, embedding)
    k = _check_k(n_neighbors, Q.shape[0] + 1)
    if k > Q.shape[0]:
        raise ValueError("n_neighbors must be at most n_samples - 1")
    return float(_qnx_all(Q)[k - 1])


def lcmc(X, embedding, n_neighbors: int = 10) -> float:
    """Local continuity meta-criterion ``Q_NX(K) - K / (n - 1)``."""
    Q = coranking_matrix(X, embedding)
    n = Q.shape[0] + 1
    k = _check_k(n_neighbors, n)
    if k > n - 1:
        raise ValueError("n_neighbors must be at most n_samples - 1")
    return float(_qnx_all(Q)[k - 1] - k / (n - 1.0))


def rnx_curve(X, embedding) -> np.ndarray:
    """Rescaled ``R_NX(K) = ((n-1) Q_NX(K) - K) / (n-1-K)`` for ``K = 1 .. n-2``."""
    Q = coranking_matrix(X, embedding)
    n = Q.shape[0] + 1
    q = _qnx_all(Q)[: n - 2]
    Ks = np.arange(1, n - 1, dtype=float)
    return ((n - 1.0) * q - Ks) / (n - 1.0 - Ks)


def rnx_auc(X, embedding) -> float:
    """Area under ``R_NX(K)`` on a log-K axis (weights ``1/K``).

    The value is 1 for a perfect embedding and about 0 for a random one.
    """
    r = rnx_curve(X, embedding)
    w = 1.0 / np.arange(1, r.size + 1, dtype=float)
    return float(np.sum(r * w) / np.sum(w))


def embedding_quality(X, embedding, n_neighbors: int = 10) -> dict:
    """All scalar metrics at one neighbourhood size, from a single ranking pass."""
    X, Y = _pair(X, embedding)
    n = X.shape[0]
    k = _check_k(n_neighbors, n)
    rx = _rank_matrix(X)
    ry = _rank_matrix(Y)
    off = ~np.eye(n, dtype=bool)
    Q = np.zeros((n - 1, n - 1), dtype=np.int64)
    np.add.at(Q, (rx[off] - 1, ry[off] - 1), 1)
    q = _qnx_all(Q)
    Ks = np.arange(1, n - 1, dtype=float)
    r = ((n - 1.0) * q[: n - 2] - Ks) / (n - 1.0 - Ks)
    w = 1.0 / Ks
    result = {
        "n_neighbors": k,
        "neighborhood_preservation": float(q[k - 1]) if k <= n - 1 else float("nan"),
        "lcmc": float(q[k - 1] - k / (n - 1.0)) if k <= n - 1 else float("nan"),
        "rnx_auc": float(np.sum(r * w) / np.sum(w)),
    }
    if k < n / 2.0:
        result["trustworthiness"] = _rank_penalty(rx, ry, k)
        result["continuity"] = _rank_penalty(ry, rx, k)
    else:
        result["trustworthiness"] = float("nan")
        result["continuity"] = float("nan")
    return result
