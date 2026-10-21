"""Linear graph embeddings: Locality Preserving Projections and NPE.

Laplacian eigenmaps (``spectral_embedding``) and LLE (``lle``) give
excellent manifold embeddings but no mapping for *new* points.  Their
linearised versions learn an explicit projection matrix ``A`` so that
``y = (x - mean) @ A`` works out of sample, like PCA, while still
preserving local neighbourhood structure instead of global variance.

* **Locality Preserving Projections** (He & Niyogi, NIPS 2003) linearise
  Laplacian eigenmaps.  With a symmetric kNN affinity ``W`` (binary or heat
  kernel), degree matrix ``D`` and Laplacian ``L = D - W``, LPP minimises
  ``sum_ij W_ij ||a'x_i - a'x_j||^2`` subject to ``a' X' D X a = 1``, i.e.
  the bottom generalised eigenvectors of ``X' L X a = lambda X' D X a``.

* **Neighborhood Preserving Embedding** (He, Cai, Yan & Zhang, ICCV 2005)
  linearises LLE.  Each point is reconstructed from its ``k`` neighbours
  with sum-to-one weights ``W``; with ``M = (I - W)'(I - W)`` NPE solves
  ``X' M X a = lambda X' X a`` for the smallest eigenvalues.

The right-hand matrices are singular when ``n_features`` exceeds the number
of samples (e.g. images), so a ridge ``reg * trace(B) / d * I`` is added and,
optionally, the data are first reduced by PCA (``pca_components``), as in
the original "Laplacianfaces" pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "LocalityPreservingProjections",
    "NeighborhoodPreservingEmbedding",
    "lpp_projection",
    "lpp_reduce",
    "npe_reduce",
]


def _as_matrix(X) -> np.ndarray:
    import pandas as pd

    matrix = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")
    return matrix


def _check_int(value, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}.")
    return int(value)


def _sq_dists(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _knn(d2: np.ndarray, k: int) -> np.ndarray:
    d = d2.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def knn_heat_affinity(X, n_neighbors: int = 5, weight: str = "heat", t: float | None = None) -> np.ndarray:
    """Symmetric kNN graph (``i ~ j`` if either is among the other's neighbours).

    ``weight="heat"`` uses ``exp(-||x_i - x_j||^2 / t)`` with ``t`` defaulting
    to the mean squared kNN distance; ``weight="binary"`` uses 1.
    """
    X = _as_matrix(X)
    n = X.shape[0]
    k = _check_int(n_neighbors, "n_neighbors", 1)
    if k >= n:
        raise ValueError("n_neighbors must be smaller than n_samples.")
    if weight not in ("heat", "binary"):
        raise ValueError("weight must be 'heat' or 'binary'.")
    d2 = _sq_dists(X)
    nbrs = _knn(d2, k)
    rows = np.repeat(np.arange(n), k)
    cols = nbrs.ravel()
    if weight == "heat":
        if t is None:
            t = float(np.mean(d2[rows, cols]))
            t = t if t > 0 else 1.0
        if t <= 0:
            raise ValueError("t must be positive.")
        vals = np.exp(-d2[rows, cols] / t)
    else:
        vals = np.ones(rows.size)
    W = np.zeros((n, n))
    W[rows, cols] = vals
    return np.maximum(W, W.T)


def _solve(A: np.ndarray, B: np.ndarray, n_components: int, reg: float):
    d = A.shape[0]
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    scale = float(np.trace(B)) / d
    B = B + (reg * scale if scale > 0 else reg) * np.eye(d)
    evals, evecs = eigh(A, B, subset_by_index=[0, n_components - 1])
    # deterministic signs: largest-magnitude loading positive
    flip = np.sign(evecs[np.argmax(np.abs(evecs), axis=0), np.arange(evecs.shape[1])])
    flip[flip == 0] = 1.0
    return evals, evecs * flip


def lpp_projection(X, W, n_components: int = 2, reg: float = 1e-9):
    """Bottom generalised eigenvectors of ``X'LX a = lambda X'DX a`` for a given affinity ``W``.

    ``X`` is used as given (no centering).  Returns ``(eigenvalues, A)`` with
    ``A`` of shape ``(n_features, n_components)``, ``A' X'DX A = I``.
    """
    X = _as_matrix(X)
    W = np.asarray(W, dtype=float)
    if W.shape != (X.shape[0], X.shape[0]):
        raise ValueError("W must be (n_samples, n_samples).")
    k = _check_int(n_components, "n_components", 1)
    if k > X.shape[1]:
        raise ValueError("n_components cannot exceed n_features.")
    deg = W.sum(axis=1)
    XD = X.T * deg
    A = XD @ X - X.T @ W @ X
    B = XD @ X
    return _solve(A, B, k, reg)


class _LinearGraphEmbedding:
    def __init__(self, n_components=2, n_neighbors=5, reg=1e-6, pca_components=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.pca_components = pca_components

    def _prepare(self, X):
        X = _as_matrix(X)
        n, d = X.shape
        k = _check_int(self.n_components, "n_components", 1)
        _check_int(self.n_neighbors, "n_neighbors", 1)
        if self.n_neighbors >= n:
            raise ValueError("n_neighbors must be smaller than n_samples.")
        if not (self.reg >= 0):
            raise ValueError("reg must be non-negative.")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.pca_components is not None:
            p = _check_int(self.pca_components, "pca_components", 1)
            p = min(p, min(n, d))
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.pca_basis_ = Vt[:p].T
        else:
            self.pca_basis_ = np.eye(d)
        Z = Xc @ self.pca_basis_
        if k > Z.shape[1]:
            raise ValueError("n_components cannot exceed the (PCA-reduced) dimension.")
        return X, Z, k

    def _finish(self, Z, evals, vecs):
        self.components_ = (self.pca_basis_ @ vecs).T  # (n_components, n_features)
        self.eigenvalues_ = evals
        self.embedding_ = Z @ vecs
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before transform.")
        X = _as_matrix(X)
        if X.shape[1] != self.mean_.size:
            raise ValueError("X has a different number of features than during fit.")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).embedding_


class LocalityPreservingProjections(_LinearGraphEmbedding):
    """LPP (He & Niyogi 2003): a linear, out-of-sample Laplacian eigenmap.

    Parameters
    ----------
    n_components, n_neighbors:
        Output dimension and kNN graph size.
    weight:
        ``"heat"`` (default) or ``"binary"`` edge weights.
    t:
        Heat-kernel width; default is the mean squared kNN distance.
    reg:
        Ridge on ``X'DX`` relative to its mean eigenvalue.
    pca_components:
        Optional PCA pre-reduction (recommended when ``n_features`` is large).

    Attributes: ``components_`` (n_components, n_features), ``eigenvalues_``
    (ascending locality cost), ``affinity_``, ``embedding_``, ``mean_``.
    """

    def __init__(self, n_components=2, n_neighbors=5, weight="heat", t=None, reg=1e-6, pca_components=None):
        super().__init__(n_components, n_neighbors, reg, pca_components)
        self.weight = weight
        self.t = t

    def fit(self, X) -> "LocalityPreservingProjections":
        X, Z, k = self._prepare(X)
        W = knn_heat_affinity(X, self.n_neighbors, self.weight, self.t)
        self.affinity_ = W
        deg = W.sum(axis=1)
        ZD = Z.T * deg
        evals, vecs = _solve(ZD @ Z - Z.T @ W @ Z, ZD @ Z, k, self.reg)
        return self._finish(Z, evals, vecs)


class NeighborhoodPreservingEmbedding(_LinearGraphEmbedding):
    """NPE (He et al. 2005): a linear, out-of-sample LLE.

    ``lle_reg`` regularises each local Gram matrix (relative to its trace)
    when solving the sum-to-one reconstruction weights.

    Attributes: ``components_``, ``eigenvalues_``, ``reconstruction_weights_``,
    ``embedding_``, ``mean_``.
    """

    def __init__(self, n_components=2, n_neighbors=10, lle_reg=1e-3, reg=1e-6, pca_components=None):
        super().__init__(n_components, n_neighbors, reg, pca_components)
        self.lle_reg = lle_reg

    def fit(self, X) -> "NeighborhoodPreservingEmbedding":
        X, Z, k = self._prepare(X)
        n = X.shape[0]
        nbrs = _knn(_sq_dists(X), self.n_neighbors)
        W = np.zeros((n, n))
        ones = np.ones(self.n_neighbors)
        for i in range(n):
            G = (X[nbrs[i]] - X[i]) @ (X[nbrs[i]] - X[i]).T
            tr = float(np.trace(G))
            G = G + (self.lle_reg * tr if tr > 0 else self.lle_reg) * np.eye(self.n_neighbors)
            w = np.linalg.solve(G, ones)
            W[i, nbrs[i]] = w / w.sum()
        self.reconstruction_weights_ = W
        IW = np.eye(n) - W
        M = IW.T @ IW
        evals, vecs = _solve(Z.T @ M @ Z, Z.T @ Z, k, self.reg)
        return self._finish(Z, evals, vecs)


def lpp_reduce(X, n_components: int = 2, n_neighbors: int = 5, **kwargs) -> np.ndarray:
    """Convenience: fit LPP and return the embedding."""
    return LocalityPreservingProjections(n_components, n_neighbors, **kwargs).fit_transform(X)


def npe_reduce(X, n_components: int = 2, n_neighbors: int = 10, **kwargs) -> np.ndarray:
    """Convenience: fit NPE and return the embedding."""
    return NeighborhoodPreservingEmbedding(n_components, n_neighbors, **kwargs).fit_transform(X)
