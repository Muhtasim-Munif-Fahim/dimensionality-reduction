"""Procrustes analysis for comparing and aligning embeddings.

Different dimensionality-reduction runs (or different methods) usually
produce embeddings that agree only up to a rotation/reflection, a global
scale and a translation. Procrustes analysis removes those nuisance
degrees of freedom before comparing two configurations.

* :func:`orthogonal_procrustes` solves ``min_R ||A R - B||_F`` subject to
  ``R^T R = I`` via the SVD of ``A^T B`` (Schönemann, 1966).
* :class:`Procrustes` fits translation + rotation (+ optional scaling) that
  maps a *source* configuration onto a *target*, and can then
  ``transform`` new points from the source space.
* :func:`procrustes_disparity` returns the standardized residual sum of
  squares (the same quantity as ``scipy.spatial.procrustes``), in
  ``[0, 1]``; 0 means identical shapes.
* :func:`protest` is Jackson's (1995) permutation test of whether two
  configurations agree more than chance (PROTEST).
* :func:`generalized_procrustes` aligns ``k >= 2`` configurations to a
  common consensus (Gower, 1975).

Configurations with fewer columns are zero-padded to the wider one, so a
2-D embedding can be compared to a 3-D one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Procrustes",
    "GeneralizedProcrustesResult",
    "ProtestResult",
    "generalized_procrustes",
    "orthogonal_procrustes",
    "procrustes_disparity",
    "protest",
]


def _as_2d(X, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _pad_pair(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"configurations need the same number of rows, got {A.shape[0]} and {B.shape[0]}"
        )
    d = max(A.shape[1], B.shape[1])
    if A.shape[1] < d:
        A = np.hstack([A, np.zeros((A.shape[0], d - A.shape[1]))])
    if B.shape[1] < d:
        B = np.hstack([B, np.zeros((B.shape[0], d - B.shape[1]))])
    return A, B


def orthogonal_procrustes(A, B, *, allow_reflection: bool = True) -> tuple[np.ndarray, float]:
    """Orthogonal matrix ``R`` minimising ``||A R - B||_F``.

    Returns ``(R, scale)`` where ``scale = trace(Sigma)`` is the sum of
    singular values of ``A^T B`` (the optimal least-squares scale factor
    is ``scale / ||A||_F^2``). With ``allow_reflection=False`` the
    solution is restricted to proper rotations (``det(R) = +1``).
    """
    A = _as_2d(A, "A")
    B = _as_2d(B, "B")
    A, B = _pad_pair(A, B)
    U, s, Vt = np.linalg.svd(A.T @ B)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        s = s.copy()
        s[-1] *= -1.0
    return U @ Vt, float(s.sum())


class Procrustes:
    """Similarity (or rigid) transform aligning a source onto a target.

    Parameters
    ----------
    scaling :
        Fit a global isotropic scale factor (similarity transform). With
        ``False`` only translation + rotation are fitted.
    allow_reflection :
        Permit improper rotations (reflections). Embeddings from PCA,
        MDS, Isomap, ... have arbitrary signs, so this defaults to True.

    Attributes
    ----------
    rotation_ :
        Orthogonal ``(d, d)`` matrix.
    scale_ :
        Scalar scale factor (1.0 when ``scaling=False``).
    translation_ :
        Vector added after rotation/scaling.
    source_mean_, target_mean_ :
        Column means of the fitted configurations.
    rmsd_ :
        Root-mean-square distance between aligned source and target rows.
    disparity_ :
        Standardized disparity in ``[0, 1]`` (see :func:`procrustes_disparity`).
    """

    def __init__(self, *, scaling: bool = True, allow_reflection: bool = True) -> None:
        self.scaling = scaling
        self.allow_reflection = allow_reflection

    def fit(self, source, target) -> "Procrustes":
        A = _as_2d(source, "source")
        B = _as_2d(target, "target")
        self._source_dim = A.shape[1]
        A, B = _pad_pair(A, B)
        if A.shape[0] < 2:
            raise ValueError("Procrustes needs at least two points")
        self.source_mean_ = A.mean(axis=0)
        self.target_mean_ = B.mean(axis=0)
        A0 = A - self.source_mean_
        B0 = B - self.target_mean_
        normA = float(np.sum(A0**2))
        if normA == 0.0:
            raise ValueError("source configuration is degenerate (all points identical)")
        R, trace = orthogonal_procrustes(A0, B0, allow_reflection=self.allow_reflection)
        self.rotation_ = R
        self.scale_ = trace / normA if self.scaling else 1.0
        self.translation_ = self.target_mean_ - self.scale_ * self.source_mean_ @ R
        aligned = self.scale_ * A0 @ R + self.target_mean_
        self.aligned_ = aligned
        self.rmsd_ = float(np.sqrt(np.mean(np.sum((aligned - B) ** 2, axis=1))))
        self.disparity_ = procrustes_disparity(A, B, allow_reflection=self.allow_reflection)
        return self

    def transform(self, X) -> np.ndarray:
        X = _as_2d(X, "X")
        if X.shape[1] != self._source_dim:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self._source_dim} (source dimension)"
            )
        d = self.rotation_.shape[0]
        if X.shape[1] < d:
            X = np.hstack([X, np.zeros((X.shape[0], d - X.shape[1]))])
        return self.scale_ * X @ self.rotation_ + self.translation_

    def fit_transform(self, source, target) -> np.ndarray:
        return self.fit(source, target).aligned_


def _standardize(X: np.ndarray) -> np.ndarray:
    X0 = X - X.mean(axis=0)
    norm = np.linalg.norm(X0)
    if norm == 0.0:
        raise ValueError("configuration is degenerate (all points identical)")
    return X0 / norm


def procrustes_disparity(A, B, *, allow_reflection: bool = True) -> float:
    """Standardized Procrustes disparity ``m^2`` in ``[0, 1]``.

    Both configurations are centred and scaled to unit Frobenius norm,
    then ``m^2 = 1 - (trace Sigma)^2``. This equals the residual sum of
    squares returned by ``scipy.spatial.procrustes``; it is symmetric in
    ``A`` and ``B``.
    """
    A = _as_2d(A, "A")
    B = _as_2d(B, "B")
    A, B = _pad_pair(A, B)
    A = _standardize(A)
    B = _standardize(B)
    _, trace = orthogonal_procrustes(A, B, allow_reflection=allow_reflection)
    return float(max(0.0, 1.0 - trace**2))


@dataclass(frozen=True)
class ProtestResult:
    """Outcome of :func:`protest`."""

    disparity: float
    correlation: float
    p_value: float
    n_permutations: int


def protest(A, B, *, n_permutations: int = 999, random_state=None) -> ProtestResult:
    """Procrustean randomization test (Jackson, 1995).

    The statistic is the Procrustes correlation ``r = sqrt(1 - m^2)``.
    Rows of ``B`` are permuted to build the null distribution; the
    p-value is ``(1 + #{r_perm >= r}) / (1 + n_permutations)``.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    A = _as_2d(A, "A")
    B = _as_2d(B, "B")
    A, B = _pad_pair(A, B)
    As = _standardize(A)
    Bs = _standardize(B)
    m2 = procrustes_disparity(As, Bs)
    r = float(np.sqrt(1.0 - m2))
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(Bs.shape[0])
        _, trace = orthogonal_procrustes(As, Bs[perm])
        if trace >= r - 1e-12:
            exceed += 1
    return ProtestResult(
        disparity=m2,
        correlation=r,
        p_value=(1 + exceed) / (1 + n_permutations),
        n_permutations=n_permutations,
    )


@dataclass(frozen=True)
class GeneralizedProcrustesResult:
    """Outcome of :func:`generalized_procrustes`."""

    aligned: list
    consensus: np.ndarray
    residual: float
    n_iter: int
    converged: bool


def generalized_procrustes(
    configurations, *, max_iter: int = 100, tol: float = 1e-8, allow_reflection: bool = True
) -> GeneralizedProcrustesResult:
    """Generalized Procrustes analysis (GPA) of ``k >= 2`` configurations.

    Every configuration is centred and scaled to unit norm, then each is
    repeatedly rotated onto the current consensus (their mean) until the
    total residual sum of squares stops decreasing by more than ``tol``.
    """
    configs = [_as_2d(c, f"configurations[{i}]") for i, c in enumerate(configurations)]
    if len(configs) < 2:
        raise ValueError("generalized_procrustes needs at least two configurations")
    n = configs[0].shape[0]
    if any(c.shape[0] != n for c in configs):
        raise ValueError("all configurations need the same number of rows")
    d = max(c.shape[1] for c in configs)
    padded = [np.hstack([c, np.zeros((n, d - c.shape[1]))]) for c in configs]
    aligned = [_standardize(c) for c in padded]
    consensus = np.mean(aligned, axis=0)
    prev = np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        aligned = [
            c @ orthogonal_procrustes(c, consensus, allow_reflection=allow_reflection)[0]
            for c in aligned
        ]
        consensus = np.mean(aligned, axis=0)
        residual = float(sum(np.sum((c - consensus) ** 2) for c in aligned))
        if prev - residual < tol:
            converged = True
            break
        prev = residual
    return GeneralizedProcrustesResult(
        aligned=aligned,
        consensus=consensus,
        residual=residual,
        n_iter=it,
        converged=converged,
    )
