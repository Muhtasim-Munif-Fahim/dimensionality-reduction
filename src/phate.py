"""PHATE — Potential of Heat-diffusion for Affinity-based Transition Embedding.

PHATE (Moon et al., 2019, *Nature Biotechnology*) is a visualisation method
built for data with continuous trajectories and branches. It combines
ideas from diffusion maps and MDS:

1. **alpha-decaying kernel.** Each point gets an adaptive bandwidth
   ``eps_k(i)`` = distance to its ``k``-th nearest neighbour, and

       K_ij = 0.5 * exp(-(d_ij / eps_k(i)) ** alpha)
            + 0.5 * exp(-(d_ij / eps_k(j)) ** alpha).

   ``alpha`` > 2 gives a sharper-than-Gaussian tail, so the affinity is
   local without a hard kNN cutoff. Entries below ``thresh`` are zeroed.
2. **Diffusion operator** ``P = D^{-1} K`` (row-stochastic), powered to
   ``P^t``. With ``t="auto"`` the diffusion time is the knee of the von
   Neumann entropy curve ``H(t) = -sum_i p_i(t) log p_i(t)``, where
   ``p_i(t)`` are the normalised ``|lambda_i|^t`` eigenvalues of ``P``.
3. **Potential distance.** Rows of ``P^t`` are mapped through the
   ``gamma``-potential (``gamma=1``: ``-log(P^t + eps)``; ``gamma=0``:
   square root; ``gamma=-1``: the raw diffusion probabilities) and the
   embedding is built from Euclidean distances between those rows.
4. **MDS.** Classical MDS on the potential distances, optionally refined
   by metric MDS (SMACOF, the default as in the reference implementation).

This is an exact dense implementation (O(n^2) memory, O(n^3) time for the
matrix power), suitable for a few thousand points. Like diffusion maps
and t-SNE it is transductive; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

from mds import (
    _as_dense,
    _as_distance_matrix,
    _classical_mds,
    _pairwise_euclidean,
    _validate_metric,
    _validate_n_components,
    _validate_random_state,
)
from smacof import SMACOF

__all__ = [
    "PHATE",
    "phate_reduce",
    "alpha_decay_kernel",
    "von_neumann_entropy",
    "find_knee",
]


class PHATE:
    """PHATE embedding via alpha-decay diffusion potentials + MDS.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    knn :
        Neighbour rank used for each point's adaptive bandwidth.
    decay :
        Kernel decay exponent ``alpha`` (``40`` in the reference
        implementation). ``None`` uses ``alpha = 2`` with the adaptive
        bandwidth, i.e. a self-tuning Gaussian kernel.
    t :
        Diffusion time (positive int) or ``"auto"`` for the von Neumann
        entropy knee.
    gamma :
        Informational distance constant in ``[-1, 1]``; ``1`` is the
        log-potential of the paper.
    mds :
        ``"metric"`` (classical MDS then SMACOF) or ``"classic"``.
    metric :
        ``"euclidean"`` or ``"precomputed"`` (``X`` is a distance matrix).
    thresh :
        Kernel entries below this value are set to zero.
    t_max :
        Largest diffusion time considered when ``t="auto"``.
    mds_max_iter :
        SMACOF iterations when ``mds="metric"``.
    random_state :
        Seed passed to SMACOF (classical initialisation is deterministic).

    Attributes
    ----------
    embedding_, kernel_, diff_op_, diff_potential_, potential_distances_,
    optimal_t_, entropy_ (``None`` unless ``t="auto"``), bandwidths_,
    n_components_, n_features_in_.
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        decay: float | None = 40.0,
        t="auto",
        gamma: float = 1.0,
        mds: str = "metric",
        metric: str = "euclidean",
        thresh: float = 1e-4,
        t_max: int = 100,
        mds_max_iter: int = 300,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.knn = _validate_positive_int(knn, "knn")
        self.decay = _validate_decay(decay)
        self.t = _validate_t(t)
        self.gamma = _validate_gamma(gamma)
        if mds not in ("metric", "classic"):
            raise ValueError('mds must be "metric" or "classic".')
        self.mds = mds
        self.metric = _validate_metric(metric)
        thresh = float(thresh)
        if not np.isfinite(thresh) or thresh < 0.0 or thresh >= 0.5:
            raise ValueError("thresh must be a finite float in [0, 0.5).")
        self.thresh = thresh
        self.t_max = _validate_positive_int(t_max, "t_max")
        self.mds_max_iter = _validate_positive_int(mds_max_iter, "mds_max_iter")
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.kernel_: np.ndarray | None = None
        self.diff_op_: np.ndarray | None = None
        self.diff_potential_: np.ndarray | None = None
        self.potential_distances_: np.ndarray | None = None
        self.optimal_t_: int | None = None
        self.entropy_: np.ndarray | None = None
        self.bandwidths_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> PHATE:
        """Compute the PHATE embedding of ``X`` and store it."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the PHATE embedding of ``X``."""
        if self.metric == "precomputed":
            distances = _as_distance_matrix(X)
            self.n_features_in_ = None
        else:
            matrix = _as_dense(X)
            if matrix.shape[0] < 2:
                raise ValueError("X must contain at least two samples.")
            self.n_features_in_ = matrix.shape[1]
            distances = _pairwise_euclidean(matrix)
        n = distances.shape[0]
        k = int(self.n_components)
        if k >= n:
            raise ValueError(f"n_components={k} must be less than n_samples={n}.")
        if self.knn >= n:
            raise ValueError(f"knn={self.knn} must be less than n_samples={n}.")

        kernel, bandwidths = alpha_decay_kernel(
            distances, knn=self.knn, decay=self.decay, thresh=self.thresh,
            precomputed=True,
        )
        diff_op = kernel / kernel.sum(axis=1, keepdims=True)

        if self.t == "auto":
            t_values, entropy = von_neumann_entropy(diff_op, t_max=self.t_max)
            t_opt = int(t_values[find_knee(entropy)])
            self.entropy_ = entropy
        else:
            t_opt = int(self.t)
            self.entropy_ = None

        diff_t = np.linalg.matrix_power(diff_op, t_opt)
        potential = _potential(diff_t, self.gamma)
        pot_dist = _pairwise_euclidean(potential)

        embedding = _classical_mds(pot_dist, k)[0]
        if self.mds == "metric":
            smacof = SMACOF(
                n_components=k, metric="precomputed", max_iter=self.mds_max_iter,
                init="classical", random_state=self.random_state,
            )
            embedding = smacof.fit_transform(pot_dist)
        embedding = _sign_flip(np.array(embedding, dtype=float))

        self.embedding_ = np.ascontiguousarray(embedding)
        self.kernel_ = kernel
        self.diff_op_ = diff_op
        self.diff_potential_ = potential
        self.potential_distances_ = pot_dist
        self.optimal_t_ = t_opt
        self.bandwidths_ = bandwidths
        self.n_components_ = k
        return self.embedding_


def phate_reduce(X, n_components: int = 2, **kwargs):
    """Fit PHATE and return a DataFrame with columns ``PHATE1``, ``PHATE2``, ..."""
    import pandas as pd

    model = PHATE(n_components=n_components, **kwargs)
    emb = model.fit_transform(X)
    columns = [f"PHATE{i + 1}" for i in range(model.n_components_)]
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(emb, index=index, columns=columns)


def alpha_decay_kernel(
    X, knn: int = 5, decay: float | None = 40.0, thresh: float = 1e-4,
    precomputed: bool = False,
):
    """Symmetric alpha-decay affinity matrix and per-point bandwidths.

    ``K_ij = 0.5 exp(-(d_ij/eps_i)^a) + 0.5 exp(-(d_ij/eps_j)^a)`` with
    ``eps_i`` the distance from ``i`` to its ``knn``-th neighbour (self
    excluded). The diagonal is 1. Returns ``(K, eps)``.
    """
    d = _as_distance_matrix(X) if precomputed else _pairwise_euclidean(_as_dense(X))
    n = d.shape[0]
    knn = _validate_positive_int(knn, "knn")
    if knn >= n:
        raise ValueError(f"knn={knn} must be less than n_samples={n}.")
    alpha = 2.0 if decay is None else _validate_decay(decay)
    # Column 0 of the sorted row is the point itself (distance 0).
    eps = np.sort(d, axis=1)[:, knn]
    positive = eps[eps > 0.0]
    floor = float(positive.min()) if positive.size else 1.0
    eps = np.where(eps > 0.0, eps, floor)
    with np.errstate(over="ignore", under="ignore"):
        a_i = np.exp(-np.power(d / eps[:, None], alpha))
        a_j = np.exp(-np.power(d / eps[None, :], alpha))
    kernel = 0.5 * (a_i + a_j)
    kernel[kernel < thresh] = 0.0
    np.fill_diagonal(kernel, 1.0)
    return kernel, eps


def von_neumann_entropy(diff_op: np.ndarray, t_max: int = 100):
    """Von Neumann entropy of ``P^t`` for ``t = 1..t_max``.

    Uses the eigenvalues of the row-stochastic operator (real because ``P``
    is similar to the symmetric ``D^{-1/2} K D^{-1/2}``). Returns
    ``(t_values, entropy)``.
    """
    p = np.asarray(diff_op, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("diff_op must be a square matrix.")
    t_max = _validate_positive_int(t_max, "t_max")
    row = p.sum(axis=1)
    if np.any(row <= 0.0) or not np.allclose(row, 1.0, atol=1e-8):
        raise ValueError("diff_op must be row-stochastic.")
    eig = np.linalg.eigvals(p)
    lam = np.abs(np.real(eig))
    t_values = np.arange(1, t_max + 1)
    entropy = np.empty(t_max)
    for idx, t in enumerate(t_values):
        powered = lam ** t
        total = powered.sum()
        prob = powered[powered > 0.0] / total
        entropy[idx] = float(-np.sum(prob * np.log(prob)))
    return t_values, entropy


def find_knee(y) -> int:
    """Index of the knee of a decreasing curve (two-segment linear fit).

    Fits a least-squares line to ``y[:i+1]`` and ``y[i:]`` for every split
    ``i`` and returns the split with the smallest total squared error, as
    in the PHATE reference implementation.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 3:
        raise ValueError("Need at least 3 points to find a knee.")
    x = np.arange(n, dtype=float)
    best, best_err = 1, np.inf
    for i in range(1, n - 1):
        err = _line_sse(x[: i + 1], y[: i + 1]) + _line_sse(x[i:], y[i:])
        if err < best_err:
            best, best_err = i, err
    return int(best)


def _line_sse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    return float(resid @ resid)


def _potential(diff_t: np.ndarray, gamma: float) -> np.ndarray:
    eps = 1e-7
    if gamma == 1.0:
        return -np.log(diff_t + eps)
    if gamma == -1.0:
        return diff_t.copy()
    c = (1.0 - gamma) / 2.0
    return np.power(np.maximum(diff_t, 0.0), c) / c


def _sign_flip(y: np.ndarray) -> np.ndarray:
    for j in range(y.shape[1]):
        col = y[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-12)
        if len(nz) and col[nz[0]] < 0.0:
            y[:, j] = -col
    return y


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_positive_int(value: object, name: str) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_decay(value: object) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("decay must be a positive finite float or None.")
    return value


def _validate_t(value: object):
    if isinstance(value, str):
        if value != "auto":
            raise ValueError('t must be a positive integer or "auto".')
        return value
    return _validate_positive_int(value, "t")


def _validate_gamma(value: object) -> float:
    value = float(value)
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("gamma must be in [-1, 1].")
    return value
