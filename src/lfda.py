"""Local Fisher Discriminant Analysis (Sugiyama, JMLR 2007).

Fisher LDA assumes every class is a single Gaussian blob. When a class is
multimodal (e.g. "disease" made of several sub-types), LDA's within-class
scatter is inflated by the distance between the modes, and the projection
squeezes those modes together -- or the ``n_classes - 1`` dimension cap hides
them altogether.

LFDA keeps the Fisher criterion but weights every pair of samples by a local
affinity ``A_ij``, so only *nearby* same-class pairs are pulled together:

* ``A_ij = exp(-||x_i - x_j||^2 / (sigma_i sigma_j))`` with the local scale
  ``sigma_i`` = distance from ``x_i`` to its ``k``-th nearest neighbour in
  its own class (Zelnik-Manor & Perona local scaling).
* local within-class weights ``W_w_ij = A_ij / n_c`` for same-class pairs,
  ``0`` otherwise;
* local between-class weights ``W_b_ij = A_ij (1/n - 1/n_c)`` for same-class
  pairs, ``1/n`` otherwise;
* ``S = 1/2 sum_ij W_ij (x_i - x_j)(x_i - x_j)' = X' (D - W) X``.

The projection solves ``S_b phi = lambda S_w phi`` for the largest
eigenvalues. Unlike LDA it can return up to ``n_features`` components. With
``affinity="none"`` (``A_ij = 1``) the scatter matrices reduce exactly to the
classical between/within scatters, so LFDA reproduces Fisher LDA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh

__all__ = ["LocalFisherDiscriminantAnalysis", "lfda_reduce"]

_EMBEDDINGS = ("weighted", "orthonormalized", "plain")
_AFFINITIES = ("local", "none")


def _as_matrix(X) -> np.ndarray:
    matrix = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values.")
    return matrix


def _sq_dists(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _local_affinity(Xc: np.ndarray, k: int) -> np.ndarray:
    """Local-scaling heat kernel inside one class."""
    m = Xc.shape[0]
    d2 = _sq_dists(Xc)
    if m == 1:
        return np.ones((1, 1))
    kk = min(k, m - 1)
    sigma = np.sqrt(np.sort(d2, axis=1)[:, kk])
    sigma = np.where(sigma > 0, sigma, 1e-12)
    return np.exp(-d2 / np.outer(sigma, sigma))


class LocalFisherDiscriminantAnalysis:
    """Supervised linear projection that preserves within-class multimodality.

    Parameters
    ----------
    n_components :
        Output dimension (``None`` -> ``n_features``).
    n_neighbors :
        ``k`` used for the local scale ``sigma_i`` (default 7, as in the paper).
    affinity :
        ``"local"`` (default) or ``"none"`` (all affinities 1 -> classical LDA).
    embedding :
        ``"weighted"`` (default; eigenvectors scaled by ``sqrt(lambda)``),
        ``"orthonormalized"`` (orthonormal basis of the same subspace) or
        ``"plain"`` (raw ``S_w``-orthonormal eigenvectors).
    reg :
        Ridge added to ``S_w`` relative to its mean eigenvalue.

    Attributes
    ----------
    components_ : (n_components, n_features) projection, ``Z = (X - mean_) @ components_.T``.
    eigenvalues_ : local Fisher ratios, descending.
    classes_, mean_, embedding_.
    """

    def __init__(
        self,
        n_components: int | None = None,
        n_neighbors: int = 7,
        affinity: str = "local",
        embedding: str = "weighted",
        reg: float = 1e-9,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.embedding = embedding
        self.reg = reg

    def _validate(self, d: int) -> int:
        if self.affinity not in _AFFINITIES:
            raise ValueError(f"affinity must be one of {_AFFINITIES}.")
        if self.embedding not in _EMBEDDINGS:
            raise ValueError(f"embedding must be one of {_EMBEDDINGS}.")
        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("n_neighbors must be an integer >= 1.")
        if not (isinstance(self.reg, (int, float)) and self.reg >= 0):
            raise ValueError("reg must be non-negative.")
        r = d if self.n_components is None else self.n_components
        if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or not 1 <= r <= d:
            raise ValueError("n_components must be an integer in [1, n_features].")
        return int(r)

    def fit(self, X, y) -> "LocalFisherDiscriminantAnalysis":
        X = _as_matrix(X)
        y = np.asarray(y)
        n, d = X.shape
        if y.shape != (n,):
            raise ValueError("y must be 1-D with one label per sample.")
        r = self._validate(d)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("LFDA needs at least two classes.")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        Ww = np.zeros((n, n))
        Wb = np.full((n, n), 1.0 / n)
        for c in self.classes_:
            idx = np.flatnonzero(y == c)
            nc = idx.size
            if self.affinity == "local":
                A = _local_affinity(Xc[idx], int(self.n_neighbors))
            else:
                A = np.ones((nc, nc))
            block = np.ix_(idx, idx)
            Ww[block] = A / nc
            Wb[block] = A * (1.0 / n - 1.0 / nc)

        def scatter(W: np.ndarray) -> np.ndarray:
            L = np.diag(W.sum(axis=1)) - W
            S = Xc.T @ L @ Xc
            return 0.5 * (S + S.T)

        Sw = scatter(Ww)
        Sb = scatter(Wb)
        scale = float(np.trace(Sw)) / d
        Sw = Sw + (self.reg * scale if scale > 0 else max(self.reg, 1e-12)) * np.eye(d)
        evals, evecs = eigh(Sb, Sw)
        order = np.argsort(evals)[::-1][:r]
        evals, evecs = evals[order], evecs[:, order]
        flip = np.sign(evecs[np.argmax(np.abs(evecs), axis=0), np.arange(r)])
        flip[flip == 0] = 1.0
        evecs = evecs * flip

        if self.embedding == "weighted":
            T = evecs * np.sqrt(np.maximum(evals, 0.0))
        elif self.embedding == "orthonormalized":
            T, _ = np.linalg.qr(evecs)
            flip = np.sign(np.sum(T * evecs, axis=0))
            flip[flip == 0] = 1.0
            T = T * flip
        else:
            T = evecs
        self.eigenvalues_ = evals
        self.components_ = T.T
        self.embedding_ = Xc @ T
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before transform.")
        X = _as_matrix(X)
        if X.shape[1] != self.mean_.size:
            raise ValueError("X has a different number of features than during fit.")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).embedding_


def lfda_reduce(X, y, n_components: int = 2, n_neighbors: int = 7) -> pd.DataFrame:
    """Convenience wrapper returning ``LFDA1..LFDAk`` columns."""
    model = LocalFisherDiscriminantAnalysis(n_components=n_components, n_neighbors=n_neighbors)
    Z = model.fit_transform(X, y)
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(Z, columns=[f"LFDA{i + 1}" for i in range(Z.shape[1])], index=index)
