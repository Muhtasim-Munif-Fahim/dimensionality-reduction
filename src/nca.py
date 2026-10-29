"""Neighbourhood Components Analysis (Goldberger, Roweis, Hinton & Salakhutdinov 2005).

NCA learns a linear map ``A`` (``d x p``) for supervised dimension reduction and
metric learning. In the embedding ``z_i = A x_i``, every point picks a neighbour
``j`` at random with the softmax probability

    p_ij = exp(-||z_i - z_j||^2) / sum_{k != i} exp(-||z_i - z_k||^2),   p_ii = 0,

and is classified correctly when that neighbour shares its label. NCA maximises
the expected number of correct leave-one-out classifications

    f(A) = sum_i p_i,   p_i = sum_{j : y_j = y_i} p_ij,

whose gradient is ``2 A sum_i (p_i sum_k p_ik x_ik x_ik' - sum_{j in C_i} p_ij x_ij x_ij')``
with ``x_ij = x_i - x_j``. With ``W_ij = p_ij [y_i = y_j] - p_i p_ij`` and ``S = W + W'``
having diagonal ``-colsum(S)``, it reduces to ``2 (Z' S X)`` with ``Z = X A'``, so one
gradient evaluation costs ``O(n^2 (p + d))``. L-BFGS (scipy) runs the optimisation.

Unlike LDA, NCA makes no Gaussian or equal-covariance assumptions. It can use more
components than ``n_classes - 1``, and since ``A`` is not constrained to be
orthogonal it also learns a scale, which is the effective softmax bandwidth.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["NeighbourhoodComponentsAnalysis", "nca_objective"]


def _as_matrix(X) -> np.ndarray:
    import pandas as pd

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy(dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    return X


def _soft_neighbours(Z):
    sq = np.sum(Z * Z, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, np.inf)
    logits = -D
    P = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    np.fill_diagonal(P, 0.0)
    return P


def nca_objective(A, X, y):
    """Return ``(f(A), grad f(A))`` for a ``(d, p)`` transform ``A``."""
    A = np.asarray(A, dtype=float)
    X = _as_matrix(X)
    labels = np.unique(np.asarray(y).ravel(), return_inverse=True)[1]
    same = labels[:, None] == labels[None, :]
    Z = X @ A.T
    P = _soft_neighbours(Z)
    Pm = P * same
    p = Pm.sum(axis=1)
    W = Pm - P * p[:, None]
    S = W + W.T
    np.fill_diagonal(S, -S.sum(axis=0))
    grad = 2.0 * (Z.T @ S @ X)
    return float(p.sum()), grad


class NeighbourhoodComponentsAnalysis:
    """Supervised linear dimension reduction by NCA.

    Parameters
    ----------
    n_components : output dimension ``d`` (default: ``n_features``).
    init : ``"auto"`` (LDA when ``d < n_classes``, else PCA), ``"pca"``, ``"lda"``,
        ``"identity"``, ``"random"`` or a ``(d, p)`` array. NCA is non-convex, and in
        high-variance noise directions PCA/random starts can stall in a poor optimum.
    max_iter, tol : L-BFGS budget and gradient tolerance.
    standardize : z-score the features first (the stored ``components_`` act on the
        raw features either way).
    random_state : seed for ``init="random"``.
    """

    def __init__(self, n_components: Optional[int] = None, init="auto", max_iter: int = 100,
                 tol: float = 1e-5, standardize: bool = False, random_state=None):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _initial(self, Xs, y, d):
        n, p = Xs.shape
        init = self.init
        if isinstance(init, str) and init == "auto":
            init = "lda" if d < np.unique(y).size else "pca"
        if isinstance(init, np.ndarray) or (not isinstance(init, str) and init is not None):
            A = np.asarray(init, dtype=float)
            if A.shape != (d, p):
                raise ValueError(f"init array must have shape ({d}, {p}).")
            return A.copy()
        if init == "identity":
            return np.eye(d, p)
        if init == "random":
            return np.random.default_rng(self.random_state).normal(size=(d, p))
        Xc = Xs - Xs.mean(axis=0)
        if init == "pca":
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            V = np.zeros((d, p))
            V[: min(d, Vt.shape[0])] = Vt[:d]
            return V
        if init == "lda":
            classes = np.unique(y)
            Sw = np.zeros((p, p))
            Sb = np.zeros((p, p))
            for c in classes:
                Xk = Xc[y == c]
                mk = Xk.mean(axis=0)
                Sw += (Xk - mk).T @ (Xk - mk)
                Sb += len(Xk) * np.outer(mk, mk)
            Sw += 1e-6 * np.trace(Sw) / p * np.eye(p)
            L = np.linalg.cholesky(Sw)
            Li = np.linalg.inv(L)
            evals, evecs = np.linalg.eigh(Li @ Sb @ Li.T)
            V = (Li.T @ evecs[:, ::-1]).T[:d]
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            return V
        raise ValueError("init must be 'auto', 'pca', 'lda', 'identity', 'random' or an array.")

    def fit(self, X, y):
        X = _as_matrix(X)
        n, p = X.shape
        y = np.asarray(y).ravel()
        if y.size != n:
            raise ValueError("X and y must have the same number of samples.")
        classes, labels = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError("NCA needs at least two classes.")
        if n < 3:
            raise ValueError("NCA needs at least three samples.")
        d = p if self.n_components is None else self.n_components
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or not 1 <= d <= p:
            raise ValueError(f"n_components must be an integer in [1, {p}].")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter must be >= 1 and tol > 0.")

        self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(p)
        scale = X.std(axis=0) if self.standardize else np.ones(p)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        Xs = (X - self.mean_) / scale
        A0 = self._initial(Xs, labels, d)
        history = []

        def fun(a):
            f, g = nca_objective(a.reshape(d, p), Xs, labels)
            history.append(f)
            return -f, -g.ravel()

        res = minimize(fun, A0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol})
        A = res.x.reshape(d, p)
        self.transform_matrix_ = A                     # acts on standardised features
        self.components_ = A / scale[None, :]          # acts on raw (centred) features
        self.objective_ = float(-res.fun)
        self.objective_history_ = np.asarray(history)
        self.n_iter_ = int(res.nit)
        self.converged_ = bool(res.success)
        self.classes_ = classes
        self.n_features_in_ = p
        self._train_embedding = Xs @ A.T
        self._train_labels = labels
        return self

    def _check(self, X):
        if not hasattr(self, "components_"):
            raise RuntimeError("NeighbourhoodComponentsAnalysis is not fitted.")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        return X

    def transform(self, X) -> np.ndarray:
        X = self._check(X)
        return ((X - self.mean_) / self.scale_) @ self.transform_matrix_.T

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def mahalanobis_matrix_(self) -> np.ndarray:
        """Learned metric ``M = A'A`` on the raw features: ``d(x, x')^2 = (x-x')' M (x-x')``."""
        return self.components_.T @ self.components_

    def leave_one_out_accuracy(self) -> float:
        """Expected stochastic-neighbour LOO accuracy on the training data, ``f(A) / n``."""
        if not hasattr(self, "components_"):
            raise RuntimeError("NeighbourhoodComponentsAnalysis is not fitted.")
        return self.objective_ / self._train_labels.size

    def predict(self, X, n_neighbors: int = 1) -> np.ndarray:
        """k-nearest-neighbour vote (Euclidean in the learned embedding) against the training set."""
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        Z = self.transform(X)
        T = self._train_embedding
        D = np.sum(Z * Z, 1)[:, None] + np.sum(T * T, 1)[None, :] - 2 * Z @ T.T
        k = min(n_neighbors, T.shape[0])
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        votes = self._train_labels[idx]
        K = self.classes_.size
        counts = np.zeros((Z.shape[0], K))
        for j in range(k):
            counts[np.arange(Z.shape[0]), votes[:, j]] += 1
        return self.classes_[np.argmax(counts, axis=1)]

    def score(self, X, y, n_neighbors: int = 1) -> float:
        return float(np.mean(self.predict(X, n_neighbors) == np.asarray(y).ravel()))
