"""Sufficient dimension reduction: Sliced Inverse Regression and SAVE.

Most methods in this kit ignore the response. Sufficient dimension
reduction looks for the fewest linear combinations ``B'x`` such that
``y`` is independent of ``x`` given ``B'x`` (the *central subspace*), for
any link ``y = f(B'x, noise)``. Both methods below work on the standardised
predictors ``z = Sigma^{-1/2} (x - mu)`` and slice the response into ``H``
groups of (nearly) equal size, or by class for a categorical ``y``.

* **SIR** (Li 1991) takes the slice means ``m_h = E[z | slice h]`` and the
  kernel matrix ``M = sum_h p_h m_h m_h'``. Its leading eigenvectors
  ``eta`` span the central subspace (under the linearity condition, e.g.
  elliptical ``x``), and ``B = Sigma^{-1/2} eta``. SIR misses directions
  where ``f`` is symmetric, because the inverse mean curve is flat there.
  Li's sequential test of ``d`` uses ``n * sum_{j>d} lambda_j``, which is
  asymptotically ``chi^2((p - d)(H - d - 1))`` for normal predictors.

* **SAVE** (Cook & Weisberg 1991) uses second moments,
  ``M = sum_h p_h (I - V_h)^2`` with ``V_h = Cov(z | slice h)``, and so
  also recovers symmetric dependence such as ``y = x_1^2 + noise``.

Directions are returned in the original ``x`` scale and normalised to unit
length, with a deterministic sign (largest-magnitude entry positive).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

__all__ = ["SlicedInverseRegression", "SAVE", "slice_response", "subspace_distance"]


def _as_matrix(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    return X


def slice_response(y, n_slices: int = 10, categorical: Optional[bool] = None) -> np.ndarray:
    """Slice labels ``0..H-1`` for ``y``.

    Continuous ``y`` is cut into ``n_slices`` groups of nearly equal size by
    rank (ties are kept together). Categorical ``y`` (auto-detected when it
    has at most ``n_slices`` distinct values) gets one slice per class.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if isinstance(n_slices, bool) or not isinstance(n_slices, (int, np.integer)) or n_slices < 2:
        raise ValueError("n_slices must be an integer >= 2.")
    uniq, inverse = np.unique(y, return_inverse=True)
    if categorical is None:
        categorical = uniq.size <= n_slices
    if categorical:
        if uniq.size < 2:
            raise ValueError("y must take at least two distinct values.")
        return inverse.astype(int)
    yf = y.astype(float)
    order = np.argsort(yf, kind="stable")
    labels = np.empty(y.size, dtype=int)
    labels[order] = np.minimum((np.arange(y.size) * n_slices) // y.size, n_slices - 1)
    # Keep tied responses in one slice (the lowest one they reach).
    for value in np.unique(yf):
        mask = yf == value
        if mask.sum() > 1:
            labels[mask] = labels[mask].min()
    _, labels = np.unique(labels, return_inverse=True)
    if labels.max() < 1:
        raise ValueError("y must take at least two distinct values.")
    return labels


def subspace_distance(A, B) -> float:
    """Frobenius distance between the orthogonal projectors onto span(A), span(B).

    0 means identical subspaces; ``sqrt(2 d)`` is the maximum for two
    ``d``-dimensional subspaces.
    """
    A = _as_matrix(A)
    B = _as_matrix(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows.")
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return float(np.linalg.norm(Qa @ Qa.T - Qb @ Qb.T))


class _SlicedBase:
    method = "base"

    def __init__(self, n_components: int = 1, n_slices: int = 10, categorical: Optional[bool] = None, reg: float = 0.0):
        self.n_components = n_components
        self.n_slices = n_slices
        self.categorical = categorical
        self.reg = reg

    def _kernel(self, Z, labels, weights) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X, y):
        X = _as_matrix(X)
        n, p = X.shape
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, (int, np.integer)) or not 1 <= self.n_components <= p:
            raise ValueError(f"n_components must be an integer in [1, {p}].")
        if self.reg < 0:
            raise ValueError("reg must be non-negative.")
        y = np.asarray(y).ravel()
        if y.size != n:
            raise ValueError("X and y must have the same number of samples.")
        if n < p + 2:
            raise ValueError("need more samples than features to whiten X.")
        labels = slice_response(y, self.n_slices, self.categorical)
        H = int(labels.max()) + 1
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / n
        if self.reg > 0:
            cov = cov + self.reg * np.trace(cov) / p * np.eye(p)
        evals, evecs = np.linalg.eigh(cov)
        if evals.min() <= 1e-12 * max(evals.max(), 1e-300):
            raise ValueError("X covariance is singular; drop collinear columns or set reg > 0.")
        inv_sqrt = evecs @ np.diag(evals ** -0.5) @ evecs.T
        Z = Xc @ inv_sqrt
        weights = np.bincount(labels, minlength=H) / n
        M = self._kernel(Z, labels, weights)
        lam, eta = np.linalg.eigh((M + M.T) / 2.0)
        order = np.argsort(lam)[::-1]
        lam, eta = np.clip(lam[order], 0.0, None), eta[:, order]
        B = inv_sqrt @ eta
        B /= np.linalg.norm(B, axis=0, keepdims=True)
        signs = np.sign(B[np.argmax(np.abs(B), axis=0), np.arange(p)])
        signs[signs == 0] = 1.0
        B *= signs
        self.n_slices_ = H
        self.slice_labels_ = labels
        self.slice_proportions_ = weights
        self.kernel_matrix_ = M
        self.eigenvalues_ = lam
        self.all_directions_ = B
        self.directions_ = B[:, : self.n_components]
        total = lam.sum()
        self.explained_ratio_ = lam / total if total > 0 else np.zeros_like(lam)
        self.n_samples_ = n
        self.n_features_in_ = p
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "directions_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        return (X - self.mean_) @ self.directions_

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)


class SlicedInverseRegression(_SlicedBase):
    """Li (1991) Sliced Inverse Regression.

    After ``fit``: ``directions_`` (p x n_components, x scale, unit norm),
    ``eigenvalues_`` of the SIR kernel, ``explained_ratio_``,
    ``all_directions_`` and the slicing details.
    """

    method = "sir"

    def _kernel(self, Z, labels, weights):
        p = Z.shape[1]
        M = np.zeros((p, p))
        for h, w in enumerate(weights):
            if w == 0:
                continue
            m = Z[labels == h].mean(axis=0)
            M += w * np.outer(m, m)
        return M

    def dimension_test(self):
        """Li's sequential chi-square tests of ``H0: d = k`` for k = 0..min(p, H-1)-1.

        Returns a list of dicts with ``d``, ``statistic``, ``df`` and
        ``p_value``; the estimated dimension is the first ``d`` not rejected.
        """
        if not hasattr(self, "eigenvalues_"):
            raise RuntimeError("SlicedInverseRegression is not fitted.")
        p, H, n = self.n_features_in_, self.n_slices_, self.n_samples_
        rows = []
        for d in range(min(p, H - 1)):
            stat = float(n * self.eigenvalues_[d:].sum())
            df = (p - d) * (H - d - 1)
            rows.append({"d": d, "statistic": stat, "df": df, "p_value": float(stats.chi2.sf(stat, df))})
        return rows

    def estimate_dimension(self, alpha: float = 0.05) -> int:
        """Smallest ``d`` whose sequential test is not rejected at ``alpha``."""
        for row in self.dimension_test():
            if row["p_value"] > alpha:
                return row["d"]
        return min(self.n_features_in_, self.n_slices_ - 1)


class SAVE(_SlicedBase):
    """Cook & Weisberg (1991) Sliced Average Variance Estimation."""

    method = "save"

    def _kernel(self, Z, labels, weights):
        p = Z.shape[1]
        M = np.zeros((p, p))
        eye = np.eye(p)
        for h, w in enumerate(weights):
            if w == 0:
                continue
            Zh = Z[labels == h]
            if Zh.shape[0] < 2:
                V = np.zeros((p, p))
            else:
                Zc = Zh - Zh.mean(axis=0)
                V = Zc.T @ Zc / Zh.shape[0]
            D = eye - V
            M += w * D @ D
        return M
