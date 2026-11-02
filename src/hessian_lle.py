"""Hessian Locally Linear Embedding in NumPy (Donoho & Grimes, 2003).

Hessian eigenmaps ("HLLE") recover the coordinates of a manifold that is
*locally isometric to an open, connected subset* of R^d. This assumption is
weaker than Isomap's convexity requirement. The method estimates, at every point,
the tangent-space Hessian of functions on the manifold, and finds the
functions whose Hessian is zero everywhere. These are exactly the affine
functions of the true isometric coordinates.

Algorithm, for ``d = n_components`` and ``k = n_neighbors``:

1. For each point, centre its ``k`` nearest neighbours and take the top-``d``
   left singular vectors ``U`` (local tangent coordinates).
2. Build ``Y = [1, U, U_a * U_b (a <= b)]``, which has ``1 + d + d(d+1)/2`` columns.
   Orthonormalise it (QR). The last ``d(d+1)/2`` columns ``W`` estimate the
   Hessian functional on the patch.
3. Accumulate ``H[patch, patch] += W W^T``.
4. The null space of ``H`` (its ``d + 1`` smallest eigenvectors) is spanned by the
   constant and the ``d`` coordinate functions. The constant is projected out,
   and an orthonormal basis of the rest, scaled so that ``Y^T Y / n = I``, is the
   embedding. This is robust to the arbitrary basis the eigensolver returns
   for the (near-)degenerate null space.

It needs ``k > d + d(d+1)/2`` neighbours. Like LLE and LTSA it is
transductive, so there is no out-of-sample ``transform``. Dense eigensolver:
``O(n^3)``, intended for up to a few thousand points.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HessianLLE", "hessian_lle_reduce"]


class HessianLLE:
    """Hessian eigenmaps / Hessian LLE.

    Parameters
    ----------
    n_components :
        Embedding dimension ``d``.
    n_neighbors :
        Neighbourhood size ``k``. It must exceed ``d + d(d+1)/2`` (5 for ``d = 2``),
        and a few more than that is recommended. If ``None``, the default is
        ``max(10, d + d(d+1)/2 + 3)``.
    reg :
        Small ridge added to ``H`` before the eigensolve.

    Attributes
    ----------
    embedding_ : (n_samples, n_components) embedding.
    hessian_matrix_ : the symmetric ``(n, n)`` Hessian quadratic form.
    eigenvalues_ : the ``d + 1`` smallest eigenvalues of ``H`` (null space incl. constant).
    reconstruction_error_ : sum of ``eigenvalues_``, the Hessian energy of the embedding.
    """

    def __init__(self, n_components: int = 2, n_neighbors: int | None = None, reg: float = 1e-12) -> None:
        if not _is_integral(n_components) or int(n_components) < 1:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        d = self.n_components
        self._dp = d * (d + 1) // 2
        if n_neighbors is None:
            n_neighbors = max(10, d + self._dp + 3)
        if not _is_integral(n_neighbors) or int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if int(n_neighbors) <= d + self._dp:
            raise ValueError(
                f"n_neighbors must be > n_components * (n_components + 3) / 2 = {d + self._dp} for Hessian LLE."
            )
        self.n_neighbors = int(n_neighbors)
        try:
            reg = float(reg)
        except (TypeError, ValueError) as exc:
            raise ValueError("reg must be a non-negative float.") from exc
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be a non-negative float.")
        self.reg = reg
        self.embedding_: np.ndarray | None = None
        self.hessian_matrix_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.reconstruction_error_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> "HessianLLE":
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        matrix = _as_dense(X)
        n, D = matrix.shape
        d, k = self.n_components, self.n_neighbors
        if k >= n:
            raise ValueError(f"n_neighbors={k} must be less than n_samples={n}.")
        if d >= D:
            raise ValueError(f"n_components={d} must be less than n_features={D}.")
        knn = _knn_indices(matrix, k)
        H = hessian_matrix(matrix, knn, d)
        if self.reg > 0:
            H = H + self.reg * np.eye(n)
        vals, vecs = np.linalg.eigh(H)
        # The null space of H is spanned by the constant vector and the d
        # coordinate functions, but eigh returns an arbitrary basis of this
        # near-degenerate space. Take all d + 1 bottom modes, project out the
        # constant, and keep the top-d orthonormal directions that remain.
        null = vecs[:, : d + 1]
        null = null - null.mean(axis=0, keepdims=True)
        U, _s, _vt = np.linalg.svd(null, full_matrices=False)
        Y = U[:, :d] * np.sqrt(n)
        for j in range(d):
            col = Y[:, j]
            nz = np.flatnonzero(np.abs(col) > 1e-12)
            if nz.size and col[nz[0]] < 0:
                Y[:, j] = -col
        self.embedding_ = np.ascontiguousarray(Y)
        self.hessian_matrix_ = H
        self.eigenvalues_ = vals[: d + 1]
        self.reconstruction_error_ = float(np.sum(vals[: d + 1]))
        self.n_features_in_ = D
        return self.embedding_


def hessian_matrix(X: np.ndarray, knn: np.ndarray, n_components: int) -> np.ndarray:
    """Assemble the HLLE quadratic form ``H = sum_i S_i^T W_i W_i^T S_i``."""
    n = X.shape[0]
    d = n_components
    dp = d * (d + 1) // 2
    H = np.zeros((n, n))
    rows, cols = np.triu_indices(d)
    for i in range(n):
        nbrs = knn[i]
        G = X[nbrs] - X[nbrs].mean(axis=0, keepdims=True)
        U = np.linalg.svd(G, full_matrices=False)[0][:, :d]
        Yi = np.hstack([np.ones((nbrs.size, 1)), U, U[:, rows] * U[:, cols]])
        Q, _ = np.linalg.qr(Yi)
        W = Q[:, d + 1 : d + 1 + dp]
        H[np.ix_(nbrs, nbrs)] += W @ W.T
    return 0.5 * (H + H.T)


def hessian_lle_reduce(X, n_components: int = 2, n_neighbors: int | None = None, reg: float = 1e-12):
    """Fit Hessian LLE and return a DataFrame with columns ``HLLE1 ..``."""
    import pandas as pd

    model = HessianLLE(n_components=n_components, n_neighbors=n_neighbors, reg=reg)
    emb = model.fit_transform(X)
    columns = [f"HLLE{i + 1}" for i in range(model.n_components)]
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(emb, index=index, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _as_dense(X) -> np.ndarray:
    import pandas as pd

    matrix = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")
    return matrix


def _knn_indices(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, -1.0)  # self first
    return np.argsort(d2, axis=1, kind="stable")[:, 1 : n_neighbors + 1]
