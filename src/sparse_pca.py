"""Sparse PCA in NumPy (Zou, Hastie & Tibshirani, 2006 style).

Ordinary PCA loadings are dense. Sparse PCA adds an L1 penalty so each
component uses only a subset of features, which helps interpretation on
wide data. This module uses a simple alternating scheme:

1. Start from the leading dense PCA directions of the centred data.
2. Soft-threshold each loading vector (L1 proximal map) and renormalise.
3. Project out the explained subspace (deflation) and repeat.

``fit`` / ``fit_transform`` / ``transform`` mirror the other ``src/``
estimators. ``sparse_pca_reduce`` returns a DataFrame embedding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SparsePCA", "sparse_pca_reduce", "soft_threshold"]


def soft_threshold(values, threshold: float) -> np.ndarray:
    """Elementwise soft-thresholding operator ``sign(x) * max(|x| - t, 0)``."""
    arr = np.asarray(values, dtype=float)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return np.sign(arr) * np.maximum(np.abs(arr) - float(threshold), 0.0)


def _as_dense(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    return arr


def _validate_n_components(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("n_components must be a positive integer")
    return int(value)


def _leading_direction(Xc: np.ndarray) -> np.ndarray:
    """First right singular vector of centred ``Xc`` (unit length)."""
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0].astype(float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        v = np.zeros(Xc.shape[1], dtype=float)
        v[0] = 1.0
        return v
    return v / norm


class SparsePCA:
    """Sparse principal components via iterative soft-thresholded loadings.

    Parameters
    ----------
    n_components :
        Number of sparse components.
    alpha :
        L1 soft-threshold level applied to each loading (larger → sparser).
    max_iter :
        Alternating soft-threshold / renormalise iterations per component.
    tol :
        Relative change in a loading vector for early stopping.
    ridge :
        Tiny ridge added when renormalising near-zero loadings.

    Attributes
    ----------
    components_ :
        Sparse loadings, shape ``(n_components, n_features)``.
    mean_ :
        Column means subtracted during ``fit``.
    n_components_ / n_features_in_ :
        Dimensions used at fit time.
    """

    def __init__(
        self,
        n_components: int = 2,
        alpha: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 1e-12,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a non-negative finite number")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite number")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be a non-negative finite number")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> SparsePCA:
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples")
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed n_features={n_features}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")

        mean = matrix.mean(axis=0)
        residual = matrix - mean
        components = np.zeros((self.n_components, n_features), dtype=float)

        for k in range(self.n_components):
            v = _leading_direction(residual)
            for _ in range(self.max_iter):
                # Soft-threshold the direction; keep it unit-norm when possible.
                v_new = soft_threshold(v, self.alpha)
                norm = float(np.linalg.norm(v_new))
                if norm <= self.ridge:
                    # Fall back to the densest coordinate of the current vector.
                    idx = int(np.argmax(np.abs(v)))
                    v_new = np.zeros_like(v)
                    v_new[idx] = 1.0 if v[idx] >= 0 else -1.0
                else:
                    v_new = v_new / norm
                if float(np.linalg.norm(v_new - v)) <= self.tol:
                    v = v_new
                    break
                v = v_new
            components[k] = v
            # Deflate: remove the projection onto this component.
            scores = residual @ v
            residual = residual - np.outer(scores, v)

        self.components_ = components
        self.mean_ = mean
        self.n_components_ = self.n_components
        self.n_features_in_ = n_features
        return self

    def transform(self, X) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("SparsePCA must be fitted before transform")
        matrix = _as_dense(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        return (matrix - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


def sparse_pca_reduce(X, n_components: int = 2, alpha: float = 0.1, **kwargs):
    """Fit SparsePCA and return a DataFrame (or ndarray) embedding."""
    model = SparsePCA(n_components=n_components, alpha=alpha, **kwargs)
    emb = model.fit_transform(X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(
            emb,
            index=X.index,
            columns=[f"SPC{i+1}" for i in range(emb.shape[1])],
        )
    return emb
