"""Sammon mapping (Sammon, 1969) — nonlinear metric MDS.

Classical MDS (Torgerson) preserves squared distances via a closed-form
eigendecomposition. Sammon mapping instead iteratively minimises a
weighted stress that emphasises relative (fractional) distance errors:

    E = (1 / sum_{i<j} d_ij) * sum_{i<j} (d_ij - ||y_i - y_j||)^2 / d_ij

so nearby points in the original space matter more than far ones. The
update is Sammon's original steepest-descent step with a magic factor
``magic`` (default 0.3) damping the second-derivative approximation.
Initialisation is classical MDS when distances are Euclidean; otherwise
a PCA projection of a dummy configuration is used.

``fit_transform`` returns the embedding. Classic Sammon mapping is
transductive; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Sammon", "sammon_reduce"]


class Sammon:
    """Sammon nonlinear mapping via steepest descent on fractional stress.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    metric :
        ``"euclidean"`` computes pairwise distances from ``X``.
        ``"precomputed"`` treats ``X`` as a square distance matrix.
    max_iter :
        Maximum number of Sammon update passes.
    tol :
        Relative stress decrease threshold for early stopping.
    magic :
        Sammon's magic factor (learning-rate damper, typically 0.3).
    init :
        ``"classical"`` seeds with classical MDS; ``"random"`` uses a
        small Gaussian cloud.
    random_state :
        Seed for random initialisation (ignored when ``init="classical"``
        on Euclidean input, which is deterministic).

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    dist_matrix_ :
        Pairwise distance matrix used for the embedding.
    stress_ :
        Final Sammon stress ``E``.
    n_iter_ :
        Number of iterations actually performed.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit`` (``None`` when
        ``metric="precomputed"``).
    reconstruction_error_ :
        Alias of ``stress_`` for pipeline / MDS API symmetry.
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "euclidean",
        max_iter: int = 300,
        tol: float = 1e-6,
        magic: float = 0.3,
        init: str = "classical",
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.metric = _validate_metric(metric)
        self.max_iter = _validate_positive_int(max_iter, "max_iter")
        self.tol = _validate_tol(tol)
        self.magic = _validate_magic(magic)
        self.init = _validate_init(init)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.stress_: float | None = None
        self.n_iter_: int | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.reconstruction_error_: float | None = None

    def fit(self, X) -> Sammon:
        """Compute a Sammon embedding of ``X`` and store it."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional Sammon embedding of ``X``."""
        k = int(self.n_components)
        if self.metric == "precomputed":
            distances = _as_distance_matrix(X)
            n_samples = distances.shape[0]
            self.n_features_in_ = None
        else:
            matrix = _as_dense(X)
            n_samples, n_features = matrix.shape
            self.n_features_in_ = n_features
            if n_samples < 2:
                raise ValueError("X must contain at least two samples.")
            distances = _pairwise_euclidean(matrix)

        if k >= n_samples:
            raise ValueError(
                f"n_components={k} must be less than n_samples={n_samples}."
            )

        # Zero distances make the 1/d_ij weight undefined; nudge them.
        d = distances.copy()
        np.fill_diagonal(d, 0.0)
        tiny = 1e-12
        off = d > 0.0
        if not np.any(off):
            raise ValueError("All pairwise distances are zero; cannot fit Sammon.")
        d = np.where(off, d, tiny)
        np.fill_diagonal(d, 0.0)

        y0 = _initial_embedding(d, k, self.init, self.random_state)
        y, stress, n_iter = _sammon_optimize(
            d, y0, max_iter=self.max_iter, tol=self.tol, magic=self.magic
        )
        # Deterministic sign flip.
        for j in range(y.shape[1]):
            col = y[:, j]
            nz = np.flatnonzero(np.abs(col) > 1e-12)
            if len(nz) and col[nz[0]] < 0.0:
                y[:, j] = -col

        self.embedding_ = np.ascontiguousarray(y)
        self.dist_matrix_ = distances
        self.stress_ = float(stress)
        self.n_iter_ = int(n_iter)
        self.n_components_ = k
        self.reconstruction_error_ = self.stress_
        return self.embedding_


def sammon_reduce(
    X,
    n_components: int = 2,
    metric: str = "euclidean",
    max_iter: int = 300,
    tol: float = 1e-6,
    magic: float = 0.3,
    init: str = "classical",
    random_state: int | None = None,
):
    """Fit Sammon mapping and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = Sammon(
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        tol=tol,
        magic=magic,
        init=init,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"SAM{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_metric(value: object) -> str:
    if value not in ("euclidean", "precomputed"):
        raise ValueError('metric must be "euclidean" or "precomputed".')
    return str(value)


def _validate_positive_int(value: object, name: str) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_tol(value: object) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tol must be a non-negative finite float.")
    return value


def _validate_magic(value: object) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("magic must be a positive finite float.")
    return value


def _validate_init(value: object) -> str:
    if value not in ("classical", "random"):
        raise ValueError('init must be "classical" or "random".')
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


def _as_distance_matrix(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "precomputed metric requires a square distance matrix "
            "of shape (n_samples, n_samples)."
        )
    if matrix.shape[0] < 2:
        raise ValueError("X must contain at least two samples.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Distance matrix must contain only finite values.")
    if np.any(matrix < 0.0):
        raise ValueError("Distance matrix must be non-negative.")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def _classical_mds(distances: np.ndarray, n_components: int) -> np.ndarray:
    n = distances.shape[0]
    d2 = distances * distances
    h = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * h @ d2 @ h
    b = 0.5 * (b + b.T)
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    positives = np.maximum(eigenvalues[:n_components], 0.0)
    return eigenvectors[:, :n_components] * np.sqrt(positives)


def _initial_embedding(
    distances: np.ndarray, n_components: int, init: str, random_state: int | None
) -> np.ndarray:
    n = distances.shape[0]
    if init == "classical":
        y = _classical_mds(distances, n_components)
        if np.all(np.abs(y) < 1e-15):
            rng = np.random.default_rng(random_state)
            y = 0.01 * rng.normal(size=(n, n_components))
        return y
    rng = np.random.default_rng(random_state)
    scale = float(np.mean(distances[np.triu_indices(n, k=1)]))
    return 0.01 * scale * rng.normal(size=(n, n_components))


def _sammon_stress(d: np.ndarray, y: np.ndarray) -> float:
    n = d.shape[0]
    dy = _pairwise_euclidean(y)
    # Upper triangle only.
    iu = np.triu_indices(n, k=1)
    dij = d[iu]
    yij = dy[iu]
    # Guard against zero distances already handled upstream.
    c = float(np.sum(dij))
    if c <= 0.0:
        return 0.0
    return float(np.sum(((dij - yij) ** 2) / dij) / c)


def _sammon_optimize(
    d: np.ndarray,
    y0: np.ndarray,
    max_iter: int,
    tol: float,
    magic: float,
):
    """Sammon steepest-descent with diagonal Hessian (Sammon 1969).

    Stress

        E = (1/c) * sum_{i<j} (d_ij - δ_ij)^2 / d_ij,   c = sum_{i<j} d_ij

    has gradient (w.r.t. low-d coordinates ``y``)

        ∂E/∂y_i = (2/c) sum_{j≠i} ((d_ij - δ_ij)/(d_ij δ_ij)) (y_i - y_j)

    and a per-coordinate second-derivative approximation

        ∂²E/∂y_iq² = (2/c) sum_{j≠i} [
            1/(d_ij δ_ij) * (
                (d_ij - δ_ij)
                - (y_iq - y_jq)^2 / δ_ij * (1 + (d_ij - δ_ij)/δ_ij)
            )
        ].

    The step is ``Δy = magic * (∂E/∂y) / |∂²E/∂y²|``. Steps that would
    increase stress are rejected and ``magic`` is halved (up to a few
    retries) so the map stays stable on easy Euclidean configurations.
    """
    y = np.array(y0, dtype=float, copy=True)
    n, kdim = y.shape
    iu = np.triu_indices(n, k=1)
    c = float(np.sum(d[iu]))
    if c <= 0.0:
        return y, 0.0, 0
    prev = _sammon_stress(d, y)
    n_iter = 0
    eps = 1e-12
    step = float(magic)
    two_over_c = 2.0 / c
    for it in range(max_iter):
        dy = _pairwise_euclidean(y)
        dy_safe = np.where(dy > eps, dy, eps)
        np.fill_diagonal(dy_safe, 1.0)

        inv_d = np.zeros_like(d)
        mask = d > eps
        inv_d[mask] = 1.0 / d[mask]
        inv_dy = 1.0 / dy_safe
        # coeff_ij = (d - δ) / (d * δ)
        coeff = (d - dy_safe) * inv_d * inv_dy
        np.fill_diagonal(coeff, 0.0)

        row_sum = coeff.sum(axis=1)
        grad = two_over_c * (row_sum[:, None] * y - coeff @ y)

        # Per-coordinate Hessian diagonal.
        qp = y[:, None, :] - y[None, :, :]  # (n, n, k)
        inv_d_dy = (inv_d * inv_dy)[:, :, None]
        term = (d - dy_safe)[:, :, None] - (qp ** 2) * (
            (1.0 + (d - dy_safe) * inv_dy)[:, :, None] * inv_dy[:, :, None]
        )
        hess = two_over_c * np.sum(inv_d_dy * term, axis=1)
        hess = np.where(np.abs(hess) < eps, eps, hess)

        proposed = y - step * grad / np.abs(hess)
        proposed -= proposed.mean(axis=0, keepdims=True)
        stress = _sammon_stress(d, proposed)

        # Backtrack if the step increased stress.
        retries = 0
        while stress > prev * (1.0 + 1e-12) and retries < 8:
            step *= 0.5
            proposed = y - step * grad / np.abs(hess)
            proposed -= proposed.mean(axis=0, keepdims=True)
            stress = _sammon_stress(d, proposed)
            retries += 1
        if stress > prev * (1.0 + 1e-12):
            # No improving step found; stop.
            break
        y = proposed
        n_iter = it + 1
        if prev > 0.0 and (prev - stress) / prev < tol:
            prev = stress
            break
        # Slowly restore step size after successful updates.
        step = min(float(magic), step * 1.05)
        prev = stress
    return y, prev, n_iter
