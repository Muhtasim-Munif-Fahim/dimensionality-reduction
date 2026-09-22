"""Truncated SVD for sparse TF-IDF-style matrices.

PCA in ``pca.py`` centers every feature. Centering fills structural zeros, so
those helpers are a poor fit for document-term matrices. This estimator factors
the matrix as stored (no centering), which is the usual latent-semantic step
on TF-IDF input.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["TruncatedSVD"]


class TruncatedSVD:
    """Leading right singular vectors of a matrix, without centering.

    Parameters
    ----------
    n_components :
        How many components to keep. An integer selects that rank. A float in
        ``(0, 1]`` selects the smallest leading rank whose explained variance
        ratio sums to at least that fraction (for example ``0.95``).

    Attributes
    ----------
    components_ :
        Right singular vectors, shape ``(n_components_, n_features)``.
    singular_values_ :
        Singular values of the fitted matrix, largest first.
    explained_variance_ :
        Column variance of the projected training scores.
    explained_variance_ratio_ :
        ``explained_variance_`` divided by the total column variance of ``X``.
    n_components_ :
        Rank actually retained after a variance threshold is applied.
    n_features_in_ :
        Number of columns seen during ``fit``.
    """

    def __init__(self, n_components: int | float = 2) -> None:
        self.n_components = _validate_n_components(n_components)
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> TruncatedSVD:
        """Factor ``X`` and store the leading singular vectors."""
        spec = _validate_n_components(self.n_components)
        matrix = _as_float_matrix(X)
        _assert_finite(matrix)
        n_samples, n_features = matrix.shape
        if n_samples < 1 or n_features < 1:
            raise ValueError("X must have at least one row and one column.")

        components, singular_values = _compact_right_vectors(matrix)
        scores = np.asarray(matrix @ components.T, dtype=np.float64)
        explained = np.var(scores, axis=0)
        total = _total_variance(matrix)
        ratios = np.zeros_like(explained) if total <= 0.0 else explained / total

        k = _select_rank(spec, ratios, n_samples, n_features)
        self.n_features_in_ = n_features
        self.n_components_ = k
        self.components_ = np.ascontiguousarray(components[:k])
        self.singular_values_ = np.ascontiguousarray(singular_values[:k])
        self.explained_variance_ = np.ascontiguousarray(explained[:k])
        self.explained_variance_ratio_ = np.ascontiguousarray(ratios[:k])
        return self

    def transform(self, X) -> np.ndarray:
        """Project ``X`` onto the fitted components. Sparse input stays sparse until the product."""
        self._check_fitted()
        matrix = _as_float_matrix(X)
        _assert_finite(matrix)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, expected {self.n_features_in_}."
            )
        return np.asarray(matrix @ self.components_.T, dtype=np.float64)

    def inverse_transform(self, X) -> np.ndarray:
        """Map scores back to the original feature space. There is no mean to add."""
        self._check_fitted()
        scores = _as_dense_2d(X)
        _assert_finite(scores)
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {scores.shape[1]} components, expected {self.n_components_}."
            )
        return scores @ self.components_

    def fit_transform(self, X) -> np.ndarray:
        """Fit on ``X`` and return the projected training scores."""
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if self.components_ is None or self.n_components_ is None:
            raise RuntimeError("This TruncatedSVD instance is not fitted yet.")


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int | float:
    if _is_integral(value):
        count = int(value)
        if count < 1:
            raise ValueError("Integer n_components must be >= 1.")
        return count
    if isinstance(value, (float, np.floating)):
        threshold = float(value)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "Float n_components must be a variance fraction in (0, 1]."
            )
        return threshold
    raise ValueError(
        "n_components must be a positive int or a float variance fraction in (0, 1]."
    )


def _select_rank(
    spec: int | float,
    ratios: np.ndarray,
    n_samples: int,
    n_features: int,
) -> int:
    limit = min(n_samples, n_features)
    if _is_integral(spec):
        rank = int(spec)
        if rank > limit:
            raise ValueError(
                f"n_components={rank} exceeds min(n_samples, n_features)={limit}."
            )
        return rank

    if ratios.size == 0:
        raise ValueError("Cannot choose a variance threshold for an empty spectrum.")
    cumulative = np.cumsum(ratios)
    index = int(np.searchsorted(cumulative, float(spec), side="left"))
    return min(max(index + 1, 1), int(ratios.size))


def _as_float_matrix(X) -> np.ndarray | sparse.spmatrix:
    if sparse.issparse(X):
        matrix = X.tocsr(copy=True)
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        return matrix
    if hasattr(X, "to_numpy"):
        array = np.asarray(X.to_numpy())
    else:
        array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    return np.asarray(array, dtype=np.float64)


def _as_dense_2d(X) -> np.ndarray:
    if sparse.issparse(X):
        array = X.toarray()
    elif hasattr(X, "to_numpy"):
        array = np.asarray(X.to_numpy())
    else:
        array = np.asarray(X)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    return array


def _assert_finite(matrix) -> None:
    data = matrix.data if sparse.issparse(matrix) else matrix
    if data.size and not np.isfinite(data).all():
        raise ValueError("Input contains NaN or infinite values.")


def _matmul_dense(left, right) -> np.ndarray:
    product = left @ right
    if sparse.issparse(product):
        product = product.toarray()
    return np.asarray(product, dtype=np.float64)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5


def _compact_right_vectors(X) -> tuple[np.ndarray, np.ndarray]:
    """Full compact SVD via the smaller Gram matrix.

    Returns every right singular vector that this Gram can recover, ordered
    from the largest singular value to the smallest. The number of vectors is
    ``min(n_samples, n_features)``.
    """
    n_samples, n_features = X.shape
    if n_features <= n_samples:
        gram = _symmetrize(_matmul_dense(X.T, X))
        eigenvalues, vectors = np.linalg.eigh(gram)
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        vectors = vectors[:, ::-1]
        return vectors.T, np.sqrt(eigenvalues)

    gram = _symmetrize(_matmul_dense(X, X.T))
    eigenvalues, left = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
    left = left[:, ::-1]
    singular = np.sqrt(eigenvalues)
    projected = _matmul_dense(X.T, left)
    components = np.zeros((n_samples, n_features), dtype=np.float64)
    scale = singular[0] if singular.size and singular[0] > 0.0 else 1.0
    stable = singular > scale * 1e-12
    if np.any(stable):
        components[stable] = (projected[:, stable] / singular[stable]).T
    return components, singular


def _total_variance(X) -> float:
    if sparse.issparse(X):
        squared = X.copy()
        squared.data = np.square(squared.data)
        mean = np.asarray(X.mean(axis=0)).ravel()
        second_moment = np.asarray(squared.mean(axis=0)).ravel()
        variance = np.maximum(second_moment - np.square(mean), 0.0)
        return float(variance.sum())
    return float(np.var(X, axis=0).sum())
