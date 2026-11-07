"""SMACOF — iterative metric MDS by majorisation (de Leeuw, 1977).

Classical MDS recovers an exact Euclidean configuration when the
input distances are Euclidean. When they are not (or when a lower
embedding dimension is forced), Kruskal stress

    sigma(Y) = sum_{i < j} (d_ij - ||y_i - y_j||)^2

has no closed form. SMACOF (Scaling by Majorizing a Complicated
Function) minimises it by repeatedly applying the Guttman transform

    Y <- V^+ B(Z) Z

where ``Z`` is the current configuration, ``B(Z)`` encodes the ratio
of target to embedded distances, and ``V^+`` is the Moore–Penrose
inverse of the Laplacian of the unit weight matrix. With unit weights
and a centred ``Z`` this reduces to ``Y = B(Z) Z / n``.

Initialisation defaults to classical MDS (Torgerson). The algorithm is
transductive; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

from mds import (
    ClassicalMDS,
    _as_dense,
    _as_distance_matrix,
    _pairwise_euclidean,
    _validate_metric,
    _validate_n_components,
    _validate_random_state,
)

__all__ = ["SMACOF", "smacof_reduce"]


class SMACOF:
    """Metric MDS via SMACOF majorisation of Kruskal raw stress.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    metric :
        ``"euclidean"`` computes pairwise distances from ``X``.
        ``"precomputed"`` treats ``X`` as a square distance matrix.
    max_iter :
        Maximum number of Guttman transform iterations.
    tol :
        Relative stress decrease ``(s_prev - s) / s_prev`` for early stop.
    init :
        ``"classical"`` seeds with classical MDS; ``"random"`` uses a
        small Gaussian cloud.
    random_state :
        Seed for random initialisation (and ignored for classical init
        on Euclidean input, which is deterministic).

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    dist_matrix_ :
        Pairwise distance matrix used for the embedding.
    stress_ :
        Final raw Kruskal stress ``sum_{i<j} (d_ij - delta_ij)^2``.
    n_iter_ :
        Number of iterations actually performed.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit`` (``None`` when
        ``metric="precomputed"``).
    reconstruction_error_ :
        Normalised stress ``sqrt(stress / sum d_ij^2)`` (MDS API alias).
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "euclidean",
        max_iter: int = 300,
        tol: float = 1e-6,
        init: str = "classical",
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.metric = _validate_metric(metric)
        self.max_iter = _validate_positive_int(max_iter, "max_iter")
        self.tol = _validate_tol(tol)
        self.init = _validate_init(init)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.stress_: float | None = None
        self.n_iter_: int | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.reconstruction_error_: float | None = None

    def fit(self, X) -> SMACOF:
        """Compute a SMACOF embedding of ``X`` and store it."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional SMACOF embedding of ``X``."""
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

        y0 = _initial_config(distances, k, self.init, self.random_state)
        embedding, stress, n_iter = _smacof(distances, y0, self.max_iter, self.tol)

        den = float(np.sum(distances ** 2))
        recon = float(np.sqrt(stress / den)) if den > 0.0 else 0.0

        self.embedding_ = np.ascontiguousarray(embedding)
        self.dist_matrix_ = distances
        self.stress_ = float(stress)
        self.n_iter_ = int(n_iter)
        self.n_components_ = k
        self.reconstruction_error_ = recon
        return self.embedding_


def smacof_reduce(
    X,
    n_components: int = 2,
    metric: str = "euclidean",
    max_iter: int = 300,
    tol: float = 1e-6,
    init: str = "classical",
    random_state: int | None = None,
):
    """Fit SMACOF and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = SMACOF(
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        tol=tol,
        init=init,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"SMACOF{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _validate_positive_int(value: object, name: str) -> int:
    if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 1):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_tol(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError("tol must be a non-negative number.")
    return float(value)


def _validate_init(value: object) -> str:
    if value not in ("classical", "random"):
        raise ValueError('init must be "classical" or "random".')
    return str(value)


def _initial_config(
    distances: np.ndarray,
    n_components: int,
    init: str,
    random_state: int | None,
) -> np.ndarray:
    n = distances.shape[0]
    if init == "classical":
        seed = ClassicalMDS(n_components=n_components, metric="precomputed")
        return seed.fit_transform(distances)
    rng = np.random.default_rng(random_state)
    y = rng.normal(scale=0.1, size=(n, n_components))
    return y - y.mean(axis=0, keepdims=True)


def _raw_stress(distances: np.ndarray, embedding: np.ndarray) -> float:
    delta = _pairwise_euclidean(embedding)
    # sum over i<j only
    iu = np.triu_indices(distances.shape[0], k=1)
    diff = distances[iu] - delta[iu]
    return float(np.dot(diff, diff))


def _guttman_transform(distances: np.ndarray, z: np.ndarray) -> np.ndarray:
    """One SMACOF Guttman update with unit weights on a centred configuration."""
    n = z.shape[0]
    delta = _pairwise_euclidean(z)
    # B_ij = -d_ij / delta_ij for i≠j when delta>0, else 0; B_ii = -sum_j B_ij
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(delta > 0.0, distances / delta, 0.0)
    np.fill_diagonal(ratio, 0.0)
    b = -ratio
    np.fill_diagonal(b, ratio.sum(axis=1))
    y = (b @ z) / n
    return y - y.mean(axis=0, keepdims=True)


def _smacof(
    distances: np.ndarray,
    y0: np.ndarray,
    max_iter: int,
    tol: float,
):
    y = y0 - y0.mean(axis=0, keepdims=True)
    stress = _raw_stress(distances, y)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        y = _guttman_transform(distances, y)
        new_stress = _raw_stress(distances, y)
        if stress > 0.0 and (stress - new_stress) / stress < tol:
            stress = new_stress
            break
        if new_stress > stress + 1e-12:
            # Numerical safeguard: keep previous if stress rose.
            break
        stress = new_stress
    # Deterministic sign flip like ClassicalMDS.
    for j in range(y.shape[1]):
        col = y[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-12)
        if len(nz) and col[nz[0]] < 0.0:
            y[:, j] = -col
    return y, stress, n_iter
