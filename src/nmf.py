"""Non-negative matrix factorization with Lee–Seung multiplicative updates.

PCA, truncated SVD, and FastICA all allow signed factors. NMF factors a
non-negative matrix as ``X ≈ W @ H`` with both factors non-negative, which is
the usual parts-based model for counts, spectra, and TF-IDF.

``W`` (the embedding) has shape ``(n_samples, n_components)`` and ``H``
(``components_``) has shape ``(n_components, n_features)``. The objective is
the mean squared residual of that product. Updates are the classic
multiplicative rules for the Frobenius norm (Lee and Seung, 2001), in NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["NMF", "nmf_reduce"]


class NMF:
    """Non-negative factors of a non-negative matrix.

    Parameters
    ----------
    n_components :
        Rank of the factorization. Must be at least 1 and at most
        ``min(n_samples, n_features)``.
    max_iter :
        Maximum multiplicative updates.
    tol :
        Stop when the relative drop in mean squared error falls to this
        value or below.
    init :
        ``"nndsvda"`` (default) is deterministic non-negative SVD with zeros
        filled by the matrix mean. ``"nndsvd"`` leaves those zeros.
        ``"random"`` draws a positive start; ``random_state`` fixes that draw.
    random_state :
        Seed for ``init="random"``. Ignored by the SVD initializations.

    Attributes
    ----------
    components_ :
        Basis ``H``, shape ``(n_components_, n_features)``.
    n_components_ :
        Rank used in the fit.
    n_features_in_ :
        Number of columns seen during ``fit``.
    n_iter_ :
        Multiplicative updates actually run.
    reconstruction_err_ :
        Mean squared residual ``mean((X - W H) ** 2)`` of the training fit.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 200,
        tol: float = 1e-4,
        init: str = "nndsvda",
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.max_iter = _validate_max_iter(max_iter)
        self.tol = _validate_tol(tol)
        self.init = _validate_init(init)
        self.random_state = _validate_random_state(random_state)
        self.components_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int | None = None
        self.reconstruction_err_: float | None = None
        self._embedding: np.ndarray | None = None

    def fit(self, X) -> NMF:
        """Factor ``X`` into non-negative ``W`` and ``H``."""
        matrix = _as_nonnegative(X)
        n_samples, n_features = matrix.shape
        k = int(self.n_components)
        limit = min(n_samples, n_features)
        if k > limit:
            raise ValueError(
                f"n_components={k} exceeds min(n_samples, n_features)={limit}."
            )

        if not np.any(matrix):
            codes = np.zeros((n_samples, k), dtype=np.float64)
            components = np.zeros((k, n_features), dtype=np.float64)
            n_iter = 0
            error = 0.0
        else:
            codes, components = _initialize(
                matrix,
                k,
                init=self.init,
                random_state=self.random_state,
            )
            codes, components, n_iter, error = _multiplicative_updates(
                matrix,
                codes,
                components,
                max_iter=self.max_iter,
                tol=self.tol,
                update_components=True,
            )

        self.n_features_in_ = n_features
        self.n_components_ = k
        self.n_iter_ = n_iter
        self.reconstruction_err_ = error
        self.components_ = np.ascontiguousarray(components)
        self._embedding = np.ascontiguousarray(codes)
        return self

    def transform(self, X) -> np.ndarray:
        """Non-negative codes for ``X`` with the fitted basis held fixed.

        Shape is ``(n_samples, n_components_)``. Training codes from
        ``fit_transform`` can differ slightly: those were updated jointly
        with the basis.
        """
        self._check_fitted()
        matrix = _as_nonnegative(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, expected {self.n_features_in_}."
            )
        if not np.any(matrix) or not np.any(self.components_):
            return np.zeros((matrix.shape[0], self.n_components_), dtype=np.float64)
        codes = _project_codes(matrix, self.components_)
        codes, _, _, _ = _multiplicative_updates(
            matrix,
            codes,
            self.components_,
            max_iter=self.max_iter,
            tol=self.tol,
            update_components=False,
        )
        return np.ascontiguousarray(codes)

    def inverse_transform(self, codes) -> np.ndarray:
        """Map codes back to feature space as ``codes @ components_``."""
        self._check_fitted()
        scores = _as_dense_codes(codes)
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"codes have {scores.shape[1]} components, expected {self.n_components_}."
            )
        return scores @ self.components_

    def fit_transform(self, X) -> np.ndarray:
        """Fit on ``X`` and return the training codes ``W``."""
        self.fit(X)
        return self._embedding.copy()

    def _check_fitted(self) -> None:
        if self.components_ is None or self._embedding is None or self.n_components_ is None:
            raise RuntimeError("This NMF instance is not fitted yet.")


def nmf_reduce(
    X,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-4,
    init: str = "nndsvda",
    random_state: int | None = None,
) -> tuple[pd.DataFrame | np.ndarray, dict]:
    """Factor ``X`` and return non-negative codes plus the basis.

    A :class:`pandas.DataFrame` input returns a ``DataFrame`` with columns
    ``NMF1``, ``NMF2``, ... and the same index. Any other input returns an
    ndarray of shape ``(n_samples, n_components)``. The info dict holds
    ``components`` (``H``), ``reconstruction_error``, and ``n_iter``.
    """
    model = NMF(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        init=init,
        random_state=random_state,
    )
    embedding = model.fit_transform(X)
    info = {
        "components": model.components_.copy(),
        "reconstruction_error": model.reconstruction_err_,
        "n_iter": model.n_iter_,
    }
    if isinstance(X, pd.DataFrame):
        columns = [f"NMF{i + 1}" for i in range(model.n_components_)]
        embedding = pd.DataFrame(embedding, index=X.index, columns=columns)
    return embedding, info


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value):
        raise ValueError("n_components must be a positive int.")
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


def _validate_init(value: object) -> str:
    if value not in ("nndsvd", "nndsvda", "random"):
        raise ValueError("init must be 'nndsvd', 'nndsvda', or 'random'.")
    return str(value)


def _as_nonnegative(X) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        array = np.asarray(X.to_numpy())
    elif hasattr(X, "toarray"):
        array = np.asarray(X.toarray())
    else:
        array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("X must have at least one row and one column.")
    if np.iscomplexobj(array):
        raise ValueError("X must be a real matrix.")
    try:
        array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a real numeric matrix.") from exc
    if array.size and not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinite values.")
    if np.any(array < 0.0):
        raise ValueError("NMF requires a non-negative matrix.")
    return array


def _as_dense_codes(codes) -> np.ndarray:
    if hasattr(codes, "to_numpy"):
        array = np.asarray(codes.to_numpy())
    elif hasattr(codes, "toarray"):
        array = np.asarray(codes.toarray())
    else:
        array = np.asarray(codes)
    if array.ndim != 2:
        raise ValueError("codes must be a 2D matrix.")
    try:
        array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("codes must be a real numeric matrix.") from exc
    if array.size and not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinite values.")
    return array


def _mean_squared_error(
    matrix: np.ndarray, codes: np.ndarray, components: np.ndarray
) -> float:
    residual = matrix - codes @ components
    return float(np.mean(residual * residual))


def _initialize(
    matrix: np.ndarray,
    n_components: int,
    init: str,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if init == "random":
        return _random_factors(matrix, n_components, random_state)
    return _nndsvd(matrix, n_components, fill_mean=init == "nndsvda")


def _random_factors(
    matrix: np.ndarray,
    n_components: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples, n_features = matrix.shape
    scale = np.sqrt(max(float(np.mean(matrix)), 1e-12) / n_components)
    codes = rng.random((n_samples, n_components)) * scale
    components = rng.random((n_components, n_features)) * scale
    return codes, components


def _nndsvd(
    matrix: np.ndarray,
    n_components: int,
    fill_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Boutsidis–Gallopoulos non-negative double SVD initialization."""
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    n_samples, n_features = matrix.shape
    codes = np.zeros((n_samples, n_components), dtype=np.float64)
    components = np.zeros((n_components, n_features), dtype=np.float64)

    codes[:, 0] = np.sqrt(singular[0]) * np.abs(left[:, 0])
    components[0, :] = np.sqrt(singular[0]) * np.abs(right[0, :])

    for j in range(1, n_components):
        x = left[:, j]
        y = right[j, :]
        x_pos = np.maximum(x, 0.0)
        x_neg = np.maximum(-x, 0.0)
        y_pos = np.maximum(y, 0.0)
        y_neg = np.maximum(-y, 0.0)
        x_pos_norm = float(np.linalg.norm(x_pos))
        x_neg_norm = float(np.linalg.norm(x_neg))
        y_pos_norm = float(np.linalg.norm(y_pos))
        y_neg_norm = float(np.linalg.norm(y_neg))
        pos_score = x_pos_norm * y_pos_norm
        neg_score = x_neg_norm * y_neg_norm
        if pos_score == 0.0 and neg_score == 0.0:
            continue
        if pos_score > neg_score:
            direction_x = x_pos / x_pos_norm
            direction_y = y_pos / y_pos_norm
            score = pos_score
        else:
            direction_x = x_neg / x_neg_norm
            direction_y = y_neg / y_neg_norm
            score = neg_score
        scale = np.sqrt(singular[j] * score)
        codes[:, j] = scale * direction_x
        components[j, :] = scale * direction_y

    codes = np.maximum(codes, 0.0)
    components = np.maximum(components, 0.0)
    if fill_mean:
        average = float(np.mean(matrix))
        if average > 0.0:
            codes[codes == 0.0] = average
            components[components == 0.0] = average
    return codes, components


def _project_codes(matrix: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Clip the unconstrained least-squares codes onto the non-negative orthant."""
    gram = components @ components.T
    rhs = matrix @ components.T
    unconstrained = rhs @ np.linalg.pinv(gram)
    codes = np.maximum(unconstrained, 0.0)
    fill = max(float(np.mean(matrix)) * 1e-3, np.finfo(np.float64).eps)
    codes[codes == 0.0] = fill
    return codes


def _multiplicative_updates(
    matrix: np.ndarray,
    codes: np.ndarray,
    components: np.ndarray,
    max_iter: int,
    tol: float,
    update_components: bool,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Lee–Seung updates. Returns the best non-negative factors seen."""
    eps = np.finfo(np.float64).eps
    codes = np.maximum(codes, 0.0).copy()
    components = np.maximum(components, 0.0).copy()
    prev_error = _mean_squared_error(matrix, codes, components)
    if not np.isfinite(prev_error):
        raise RuntimeError("NMF updates produced a non-finite reconstruction error.")
    best_error = prev_error
    best_codes = codes.copy()
    best_components = components.copy()
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        if update_components:
            numer = codes.T @ matrix
            denom = (codes.T @ codes) @ components
            components *= numer / np.maximum(denom, eps)
            components = np.maximum(components, 0.0)

        numer = matrix @ components.T
        denom = codes @ (components @ components.T)
        codes *= numer / np.maximum(denom, eps)
        codes = np.maximum(codes, 0.0)

        if update_components:
            scales = np.sum(components, axis=1)
            alive = scales > eps
            if np.any(alive):
                components[alive] /= scales[alive, None]
                codes[:, alive] *= scales[alive]

        error = _mean_squared_error(matrix, codes, components)
        if not np.isfinite(error):
            raise RuntimeError("NMF updates produced a non-finite reconstruction error.")
        if error < best_error:
            best_error = error
            best_codes = codes.copy()
            best_components = components.copy()
        improvement = (prev_error - error) / max(prev_error, eps)
        if improvement <= tol:
            break
        prev_error = error

    return best_codes, best_components, n_iter, float(best_error)
