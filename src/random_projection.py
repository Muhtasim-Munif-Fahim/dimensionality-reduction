"""Gaussian and sparse random projections (Johnson-Lindenstrauss).

The Johnson-Lindenstrauss lemma says that ``n`` points can be mapped into
``k = O(log n / eps^2)`` dimensions by a random linear map while every
pairwise squared distance is kept within a factor ``(1 +/- eps)`` with high
probability. The map needs no fitting to the data, so it is a cheap
preprocessing step before kNN, clustering or a slower nonlinear method.

* :func:`johnson_lindenstrauss_min_dim` -- the safe target dimension
  ``k >= 4 ln(n) / (eps^2 / 2 - eps^3 / 3)`` (Dasgupta & Gupta, 2003).
* :class:`GaussianRandomProjection` -- i.i.d. ``N(0, 1/k)`` entries.
* :class:`SparseRandomProjection` -- Achlioptas (2003) / Li, Hastie &
  Church (2006) very sparse projections. Each entry is
  ``+/- sqrt(1 / (density * k))`` with probability ``density / 2`` each and
  0 otherwise. ``density="auto"`` uses ``1 / sqrt(n_features)``, and
  ``density=1/3`` is the original Achlioptas scheme. The matrix is
  stored as SciPy CSR, so projecting sparse TF-IDF input stays cheap.
* :func:`distance_distortion` -- empirical ratios of projected to original
  pairwise squared distances, to check the guarantee on your data.

Both transformers scale entries so that ``E ||R x||^2 = ||x||^2``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

__all__ = [
    "GaussianRandomProjection",
    "SparseRandomProjection",
    "distance_distortion",
    "johnson_lindenstrauss_min_dim",
]


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """Smallest safe target dimension for ``n_samples`` points at distortion ``eps``.

    ``n_samples`` and ``eps`` may be scalars or arrays (broadcast together);
    a scalar pair returns an ``int``.
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    n_arr = np.asarray(n_samples, dtype=np.float64)
    if np.any(eps_arr <= 0.0) or np.any(eps_arr >= 1.0):
        raise ValueError("eps must be in (0, 1).")
    if np.any(n_arr < 1):
        raise ValueError("n_samples must be >= 1.")
    denom = eps_arr**2 / 2.0 - eps_arr**3 / 3.0
    k = np.ceil(4.0 * np.log(n_arr) / denom).astype(np.int64)
    k = np.maximum(k, 1)
    if k.ndim == 0:
        return int(k)
    return k


def _as_matrix(X):
    if sparse.issparse(X):
        matrix = sparse.csr_matrix(X, dtype=np.float64)
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("X contains NaN or infinite values.")
        return matrix
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X contains NaN or infinite values.")
    return matrix


def _resolve_components(n_components, eps, n_samples, n_features):
    if isinstance(n_components, str):
        if n_components != "auto":
            raise ValueError("n_components must be a positive integer or 'auto'.")
        k = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        if k > n_features:
            raise ValueError(
                f"eps={eps} needs n_components={k} for {n_samples} samples, more than "
                f"n_features={n_features}; raise eps or pass an explicit n_components."
            )
        return int(k)
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError("n_components must be a positive integer or 'auto'.")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    return int(n_components)


class _BaseRandomProjection:
    def __init__(self, n_components="auto", eps=0.1, random_state=None):
        if not 0.0 < float(eps) < 1.0:
            raise ValueError("eps must be in (0, 1).")
        self.n_components = n_components
        self.eps = float(eps)
        self.random_state = random_state
        self.components_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    def _make_components(self, k, d, rng):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X, y=None):
        """Draw the random matrix; only the shape of ``X`` is used."""
        matrix = _as_matrix(X)
        n_samples, n_features = matrix.shape
        if n_samples < 1 or n_features < 1:
            raise ValueError("X must have at least one row and one column.")
        k = _resolve_components(self.n_components, self.eps, n_samples, n_features)
        rng = np.random.default_rng(self.random_state)
        self.components_ = self._make_components(k, n_features, rng)
        self.n_components_ = k
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Project ``X`` to ``n_components_`` dimensions."""
        if self.components_ is None:
            raise RuntimeError(f"This {type(self).__name__} instance is not fitted yet.")
        matrix = _as_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, expected {self.n_features_in_}."
            )
        out = matrix @ self.components_.T
        return self._finalize(out)

    def _finalize(self, out):
        if sparse.issparse(out):
            return out.toarray()
        return np.asarray(out, dtype=np.float64)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Y):
        """Least-squares map back via the pseudo-inverse of the projection.

        Exact only on the row space of the projection; with ``k < d`` the
        information outside it is lost.
        """
        if self.components_ is None:
            raise RuntimeError(f"This {type(self).__name__} instance is not fitted yet.")
        if sparse.issparse(Y):
            Y = Y.toarray()
        scores = np.asarray(Y, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError(f"Y must have shape (n, {self.n_components_}).")
        R = self.components_.toarray() if sparse.issparse(self.components_) else self.components_
        return scores @ np.linalg.pinv(R).T


class GaussianRandomProjection(_BaseRandomProjection):
    """Dense projection with i.i.d. ``N(0, 1 / n_components)`` entries.

    Parameters
    ----------
    n_components :
        Target dimension, or ``"auto"`` for :func:`johnson_lindenstrauss_min_dim`
        of the number of training rows at ``eps``.
    eps :
        JL distortion used when ``n_components="auto"``.
    random_state :
        Seed for the random matrix.
    """

    def _make_components(self, k, d, rng):
        return rng.normal(0.0, 1.0 / math.sqrt(k), size=(k, d))


class SparseRandomProjection(_BaseRandomProjection):
    """Achlioptas / Li et al. very sparse random projection.

    Parameters
    ----------
    n_components, eps, random_state :
        As in :class:`GaussianRandomProjection`.
    density :
        Fraction of non-zero entries in ``(0, 1]``, or ``"auto"`` for
        ``1 / sqrt(n_features)``. ``1/3`` reproduces Achlioptas (2003).
    dense_output :
        Return a dense array (default). With ``False`` and sparse input
        the projection is returned as CSR.

    Attributes
    ----------
    density_ :
        Density actually used.
    components_ :
        CSR matrix of shape ``(n_components_, n_features)``.
    """

    def __init__(self, n_components="auto", density="auto", eps=0.1, dense_output=True, random_state=None):
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)
        if isinstance(density, str):
            if density != "auto":
                raise ValueError("density must be in (0, 1] or 'auto'.")
        elif not 0.0 < float(density) <= 1.0:
            raise ValueError("density must be in (0, 1] or 'auto'.")
        self.density = density
        self.dense_output = bool(dense_output)
        self.density_ = None

    def _make_components(self, k, d, rng):
        density = 1.0 / math.sqrt(d) if self.density == "auto" else float(self.density)
        self.density_ = density
        rows, cols, vals = [], [], []
        scale = math.sqrt(1.0 / (density * k))
        for i in range(k):
            nnz = int(rng.binomial(d, density))
            if nnz == 0:
                continue
            idx = rng.choice(d, size=nnz, replace=False)
            signs = rng.integers(0, 2, size=nnz) * 2 - 1
            rows.append(np.full(nnz, i))
            cols.append(idx)
            vals.append(signs * scale)
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            v = np.concatenate(vals).astype(np.float64)
        else:
            r = c = np.zeros(0, dtype=np.int64)
            v = np.zeros(0)
        return sparse.csr_matrix((v, (r, c)), shape=(k, d))

    def _finalize(self, out):
        if sparse.issparse(out) and not self.dense_output:
            return sparse.csr_matrix(out)
        return super()._finalize(out)


def distance_distortion(X, Y, n_pairs=None, random_state=None):
    """Ratios ``||y_i - y_j||^2 / ||x_i - x_j||^2`` over distinct pairs.

    ``n_pairs`` (optional) samples that many random pairs instead of all
    ``n (n - 1) / 2``. Pairs with zero original distance are skipped.
    Returns a dict with ``ratios``, ``min``, ``max``, ``mean`` and
    ``max_abs_distortion = max |ratio - 1|``.
    """
    A = _as_matrix(X)
    B = _as_matrix(Y)
    A = A.toarray() if sparse.issparse(A) else A
    B = B.toarray() if sparse.issparse(B) else B
    if A.shape[0] != B.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    n = A.shape[0]
    if n < 2:
        raise ValueError("need at least two rows.")
    if n_pairs is None:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(random_state)
        i = rng.integers(0, n, size=int(n_pairs))
        j = rng.integers(0, n, size=int(n_pairs))
        keep = i != j
        i, j = i[keep], j[keep]
    d_orig = np.sum((A[i] - A[j]) ** 2, axis=1)
    d_proj = np.sum((B[i] - B[j]) ** 2, axis=1)
    ok = d_orig > 0
    ratios = d_proj[ok] / d_orig[ok]
    if ratios.size == 0:
        raise ValueError("all sampled pairs have zero original distance.")
    return {
        "ratios": ratios,
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "mean": float(ratios.mean()),
        "max_abs_distortion": float(np.max(np.abs(ratios - 1.0))),
    }
