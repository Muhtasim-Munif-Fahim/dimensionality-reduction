"""Landmark MDS (de Silva & Tenenbaum, 2004).

Classical MDS needs the full ``n x n`` distance matrix and an ``O(n^3)``
eigendecomposition. Landmark MDS picks ``m << n`` landmark points, runs
classical MDS on the ``m x m`` landmark distances only, and then places every
other point by distance-based triangulation against the landmarks:

``y(x) = -1/2 * L_pinv @ (delta_x - delta_mu)``

where ``delta_x`` holds the squared distances from ``x`` to each landmark,
``delta_mu`` is the column mean of the squared landmark distance matrix, and
``L_pinv`` has rows ``v_i / sqrt(lambda_i)`` built from the leading
eigenpairs of the double-centred landmark Gram matrix. The cost is
``O(n m)`` distances plus an ``O(m^3)`` eigendecomposition, and the same
formula gives a genuine out-of-sample ``transform``.

When the data are Euclidean of rank ``k`` and the landmarks span that
``k``-dimensional affine hull, the embedding matches classical MDS up to a
rigid motion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["LandmarkMDS", "landmark_mds_reduce"]

_SELECTIONS = ("maxmin", "random")


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sq = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(sq, 0.0)


class LandmarkMDS:
    """Scalable metric MDS through landmark triangulation.

    Parameters
    ----------
    n_components :
        Embedding dimension.
    n_landmarks :
        Number of landmarks ``m`` (clipped to ``n_samples``). Must be at
        least ``n_components + 1``.
    selection :
        ``"maxmin"`` (greedy farthest-point, well spread; default) or
        ``"random"`` (uniform without replacement).
    random_state :
        Seed for the first maxmin landmark / random selection.

    Attributes
    ----------
    landmarks_ :
        Landmark coordinates in input space, shape ``(m, n_features)``.
    landmark_indices_ :
        Row indices of the landmarks in the fitted ``X``.
    embedding_ :
        Embedding of the fitted data, shape ``(n_samples, n_components_)``.
    eigenvalues_ :
        Leading eigenvalues of the landmark Gram matrix.
    n_components_ :
        Embedding dimension actually used (positive eigenvalues only).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_landmarks: int = 50,
        selection: str = "maxmin",
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if isinstance(n_landmarks, bool) or not isinstance(n_landmarks, (int, np.integer)) or n_landmarks < 2:
            raise ValueError("n_landmarks must be an integer >= 2")
        if n_landmarks <= n_components:
            raise ValueError("n_landmarks must exceed n_components")
        if selection not in _SELECTIONS:
            raise ValueError(f"selection must be one of {_SELECTIONS}")
        self.n_components = int(n_components)
        self.n_landmarks = int(n_landmarks)
        self.selection = selection
        self.random_state = random_state
        self.landmarks_: np.ndarray | None = None
        self.landmark_indices_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self._pinv: np.ndarray | None = None
        self._delta_mu: np.ndarray | None = None

    def _select(self, X: np.ndarray, m: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if self.selection == "random":
            return np.sort(rng.choice(n, size=m, replace=False))
        chosen = [int(rng.integers(n))]
        min_d = _sq_dists(X, X[chosen])[:, 0]
        for _ in range(m - 1):
            nxt = int(np.argmax(min_d))
            chosen.append(nxt)
            min_d = np.minimum(min_d, _sq_dists(X, X[[nxt]])[:, 0])
        return np.asarray(chosen)

    def fit(self, X) -> "LandmarkMDS":
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        n = X.shape[0]
        m = min(self.n_landmarks, n)
        if m <= self.n_components:
            raise ValueError("need more samples than n_components")
        idx = self._select(X, m)
        L = X[idx]
        delta = _sq_dists(L, L)
        H = np.eye(m) - np.full((m, m), 1.0 / m)
        B = -0.5 * H @ delta @ H
        B = (B + B.T) / 2.0
        vals, vecs = np.linalg.eigh(B)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        tol = max(vals[0], 0.0) * 1e-10 + 1e-12
        k = int(min(self.n_components, np.sum(vals > tol)))
        if k < 1:
            raise ValueError("landmark Gram matrix has no positive eigenvalues")
        vals, vecs = vals[:k], vecs[:, :k]
        self._pinv = (vecs / np.sqrt(vals)).T
        self._delta_mu = delta.mean(axis=0)
        self.landmarks_ = L
        self.landmark_indices_ = idx
        self.eigenvalues_ = vals
        self.n_components_ = k
        self.n_features_in_ = X.shape[1]
        self.embedding_ = self.transform(X)
        return self.embedding_

    def transform(self, X) -> np.ndarray:
        """Place new points by triangulation against the fitted landmarks."""
        if self._pinv is None:
            raise RuntimeError("LandmarkMDS is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        delta_x = _sq_dists(X, self.landmarks_)
        return -0.5 * (delta_x - self._delta_mu) @ self._pinv.T


def landmark_mds_reduce(
    X,
    n_components: int = 2,
    n_landmarks: int = 50,
    selection: str = "maxmin",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return the Landmark MDS embedding as a DataFrame (``LMDS1``, ``LMDS2``, ...)."""
    index = X.index if isinstance(X, pd.DataFrame) else None
    model = LandmarkMDS(n_components, n_landmarks, selection, random_state)
    emb = model.fit_transform(X)
    return pd.DataFrame(emb, columns=[f"LMDS{i + 1}" for i in range(emb.shape[1])], index=index)
