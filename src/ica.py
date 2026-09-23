"""FastICA for linear mixtures of independent sources.

PCA and truncated SVD recover orthogonal directions of variance. Independent
component analysis instead rotates a whitened subspace so the coordinates are
as statistically independent as a fixed-point search can make them.

The model is ``X = S @ mixing_.T + mean_``. ``S`` and the columns of
``mixing_`` are identifiable only up to sign and order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FastICA"]


class FastICA:
    """Symmetric FastICA with the logcosh contrast (Hyvärinen 1999).

    Parameters
    ----------
    n_components :
        Number of independent sources to estimate. ``None`` keeps every
        feature, which requires ``n_samples >= n_features``.
    max_iter :
        Maximum symmetric fixed-point iterations.
    tol :
        Stop when the maximum absolute change in ``|diag(W W_old.T)|`` drops
        below this value.
    random_state :
        Seed for the initial unmixing matrix. An integer makes ``fit``
        reproducible.

    Attributes
    ----------
    components_ :
        Unmixing matrix, shape ``(n_components_, n_features)``.
    mixing_ :
        Least-squares mixing matrix, shape ``(n_features, n_components_)``.
    mean_ :
        Feature means subtracted during ``fit``.
    n_components_ :
        Number of sources retained.
    n_features_in_ :
        Number of columns seen during ``fit``.
    n_iter_ :
        Fixed-point iterations actually run.
    """

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.max_iter = _validate_max_iter(max_iter)
        self.tol = _validate_tol(tol)
        self.random_state = _validate_random_state(random_state)
        self.components_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None

    def fit(self, X) -> FastICA:
        """Whiten ``X`` and estimate an unmixing matrix."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")

        k = n_features if self.n_components is None else int(self.n_components)
        limit = min(n_samples, n_features)
        if k > limit:
            raise ValueError(
                f"n_components={k} exceeds min(n_samples, n_features)={limit}."
            )

        mean = matrix.mean(axis=0)
        centered = matrix - mean
        whitening = _whitening_matrix(centered, k)
        white = centered @ whitening.T
        unmixing, n_iter = _symmetric_fastica(
            white.T,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        components = unmixing @ whitening
        self.mean_ = mean
        self.n_features_in_ = n_features
        self.n_components_ = k
        self.n_iter_ = n_iter
        self.components_ = np.ascontiguousarray(components)
        self.mixing_ = np.ascontiguousarray(np.linalg.pinv(components))
        return self

    def transform(self, X) -> np.ndarray:
        """Return estimated sources for ``X``, shape ``(n_samples, n_components_)``."""
        self._check_fitted()
        matrix = _as_dense(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, expected {self.n_features_in_}."
            )
        return (matrix - self.mean_) @ self.components_.T

    def inverse_transform(self, sources) -> np.ndarray:
        """Map estimated sources back to the original feature space."""
        self._check_fitted()
        scores = _as_dense(sources)
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"sources have {scores.shape[1]} components, expected {self.n_components_}."
            )
        return scores @ self.mixing_.T + self.mean_

    def fit_transform(self, X) -> np.ndarray:
        """Fit on ``X`` and return the estimated training sources."""
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if self.components_ is None or self.mixing_ is None or self.mean_ is None:
            raise RuntimeError("This FastICA instance is not fitted yet.")


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int | None:
    if value is None:
        return None
    if not _is_integral(value):
        raise ValueError("n_components must be a positive int or None.")
    count = int(value)
    if count < 1:
        raise ValueError("n_components must be >= 1.")
    return count


def _validate_max_iter(value: object) -> int:
    if not _is_integral(value):
        raise ValueError("max_iter must be a positive int.")
    count = int(value)
    if count < 1:
        raise ValueError("max_iter must be >= 1.")
    return count


def _validate_tol(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError("tol must be a positive float.")
    tol = float(value)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive float.")
    return tol


def _validate_random_state(value: object) -> int | None:
    if value is None:
        return None
    if not _is_integral(value):
        raise ValueError("random_state must be an int or None.")
    return int(value)


def _as_dense(X) -> np.ndarray:
    if hasattr(X, "tocoo") and not hasattr(X, "to_numpy"):
        raise ValueError("FastICA expects a dense array or DataFrame, not a sparse matrix.")
    if hasattr(X, "to_numpy"):
        array = np.asarray(X.to_numpy())
    else:
        array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    array = np.asarray(array, dtype=np.float64)
    if array.size and not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinite values.")
    return array


def _whitening_matrix(centered: np.ndarray, n_components: int) -> np.ndarray:
    """Return ``K`` such that ``centered @ K.T`` has identity covariance.

    ``K`` has shape ``(n_components, n_features)`` and spans the leading
    principal subspace. Directions with no variance cannot be whitened.
    """
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    n_samples = centered.shape[0]
    variances = (singular ** 2) / n_samples
    if singular.size == 0 or variances[0] <= 0.0:
        raise ValueError("X has no variance after centering.")
    kept = variances[:n_components]
    if np.any(kept <= variances[0] * 1e-12):
        raise ValueError(
            "n_components includes a near-zero variance direction; reduce n_components."
        )
    return vt[:n_components] / np.sqrt(kept)[:, None]


def _symmetric_decorrelation(weights: np.ndarray) -> np.ndarray:
    """Project ``weights`` so its rows are orthonormal."""
    gram = weights @ weights.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    scaling = eigenvectors * (1.0 / np.sqrt(eigenvalues))
    return (scaling @ eigenvectors.T) @ weights


def _symmetric_fastica(
    white: np.ndarray,
    max_iter: int,
    tol: float,
    random_state: int | None,
) -> tuple[np.ndarray, int]:
    """Run symmetric FastICA on whitened data of shape ``(n_components, n_samples)``."""
    n_components, n_samples = white.shape
    rng = np.random.default_rng(random_state)
    initial = rng.standard_normal((n_components, n_components))
    weights = _symmetric_decorrelation(initial)

    for iteration in range(1, max_iter + 1):
        projected = weights @ white
        response = np.tanh(projected)
        derivative = np.mean(1.0 - response * response, axis=1)
        updated = (response @ white.T) / n_samples - derivative[:, None] * weights
        updated = _symmetric_decorrelation(updated)
        change = np.max(np.abs(np.abs(np.diag(updated @ weights.T)) - 1.0))
        weights = updated
        if change < tol:
            return weights, iteration
    return weights, max_iter
