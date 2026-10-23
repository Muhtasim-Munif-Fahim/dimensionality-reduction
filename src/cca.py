"""Two-view linear methods: CCA, PLS-SVD and PLS regression.

Every other method in this kit reduces one table ``X``.  When the same
samples are measured in two views ``X`` (n x p) and ``Y`` (n x q), for
example gene expression and phenotypes, we look for paired directions
``a``, ``b`` whose projections ``Xa`` and ``Yb`` move together.

* **Canonical Correlation Analysis** (Hotelling 1936) maximises
  ``corr(Xa, Yb)``.  With centred covariances ``Cxx``, ``Cyy`` and ``Cxy``,
  whiten each view with ``Cxx^{-1/2}`` and ``Cyy^{-1/2}`` and take the SVD of
  ``Cxx^{-1/2} Cxy Cyy^{-1/2} = U S V'``.  The singular values are the
  canonical correlations, and ``A = Cxx^{-1/2} U`` and ``B = Cyy^{-1/2} V``.
  Each view's canonical variates are uncorrelated with unit variance.  A
  ridge ``reg * trace(C) / d * I`` (regularised CCA, Vinod 1976) keeps the
  problem well posed when ``p`` or ``q`` approaches ``n``.

* **PLS-SVD** (Tucker's inter-battery analysis) maximises the covariance
  ``cov(Xa, Yb)`` with ``||a|| = ||b|| = 1``.  The answer is the singular
  vectors of ``X'Y``, with no whitening.

* **PLS regression** (Wold's PLS2, the regression mode of NIPALS) extracts
  latent scores ``t = X_k w``, where ``w`` is the dominant left singular
  vector of the deflated cross-product ``X_k' Y_k``.  It regresses both blocks on ``t``
  and deflates them.  The coefficients ``W (P'W)^{-1} Q'`` give a
  shrunken, low-rank least-squares fit.  The dominant singular vector is computed exactly
  with an SVD instead of power iteration.  With one response, this is PLS1.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CCA",
    "PLSSVD",
    "PLSRegression",
    "canonical_correlations",
]


def _as_matrix(X, name: str = "X") -> np.ndarray:
    import pandas as pd

    if isinstance(X, (pd.DataFrame, pd.Series)):
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _check_int(value, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}.")
    return int(value)


def _check_pair(X, Y):
    X = _as_matrix(X, "X")
    Y = _as_matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    return X, Y


def _center_scale(M: np.ndarray, scale: bool):
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1) if scale else np.ones(M.shape[1])
    std = np.where(std > 0, std, 1.0)
    return (M - mean) / std, mean, std


def _inv_sqrt(C: np.ndarray, reg: float) -> np.ndarray:
    d = C.shape[0]
    C = 0.5 * (C + C.T)
    scale = float(np.trace(C)) / d
    C = C + (reg * scale if scale > 0 else reg) * np.eye(d)
    evals, evecs = np.linalg.eigh(C)
    tol = max(evals.max(), 0.0) * d * np.finfo(float).eps
    if evals.min() <= tol:
        raise ValueError("Covariance is singular; increase reg or reduce the number of features.")
    return (evecs / np.sqrt(evals)) @ evecs.T


def _flip_pair(U: np.ndarray, V: np.ndarray):
    """Deterministic signs: largest-magnitude entry of each U column positive."""
    signs = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs, V * signs


class CCA:
    """Canonical Correlation Analysis, solved in closed form by whitening and an SVD.

    Parameters
    ----------
    n_components : number of canonical pairs (at most ``min(p, q)``).
    reg : ridge added to each covariance, relative to its mean eigenvalue
        (``0`` gives classical CCA).
    scale : standardise each column before fitting.  Canonical correlations
        don't depend on this, but the weights do.

    Attributes
    ----------
    correlations_ : canonical correlations in descending order.
    x_weights_, y_weights_ : ``A`` (p x k) and ``B`` (q x k).  ``X_c A`` and
        ``Y_c B`` have unit variance (when ``reg=0``).
    x_loadings_, y_loadings_ : correlations between each original variable and the
        canonical variates of its own view (structure coefficients).
    """

    def __init__(self, n_components: int = 2, reg: float = 0.0, scale: bool = False):
        self.n_components = n_components
        self.reg = reg
        self.scale = scale

    def fit(self, X, Y):
        k = _check_int(self.n_components, "n_components", 1)
        if not np.isfinite(self.reg) or self.reg < 0:
            raise ValueError("reg must be a non-negative finite number.")
        X, Y = _check_pair(X, Y)
        if k > min(X.shape[1], Y.shape[1]):
            raise ValueError("n_components must be <= min(n_features_X, n_features_Y).")
        n = X.shape[0]
        Xc, self.x_mean_, self.x_std_ = _center_scale(X, self.scale)
        Yc, self.y_mean_, self.y_std_ = _center_scale(Y, self.scale)
        Cxx = Xc.T @ Xc / (n - 1)
        Cyy = Yc.T @ Yc / (n - 1)
        Cxy = Xc.T @ Yc / (n - 1)
        Wx = _inv_sqrt(Cxx, self.reg)
        Wy = _inv_sqrt(Cyy, self.reg)
        U, s, Vt = np.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
        A = Wx @ U[:, :k]
        B = Wy @ Vt[:k].T
        A, B = _flip_pair(A, B)
        self.x_weights_ = A
        self.y_weights_ = B
        self.correlations_ = np.clip(s[:k], 0.0, 1.0)
        Tx, Ty = Xc @ A, Yc @ B
        self.x_loadings_ = _corr_cols(Xc, Tx)
        self.y_loadings_ = _corr_cols(Yc, Ty)
        self.n_features_in_ = (X.shape[1], Y.shape[1])
        return self

    def _check_fitted(self):
        if not hasattr(self, "x_weights_"):
            raise RuntimeError("CCA instance is not fitted yet; call fit first.")

    def transform(self, X, Y=None):
        """Canonical variates ``X_c A`` (and ``Y_c B`` if ``Y`` is given)."""
        self._check_fitted()
        X = _as_matrix(X, "X")
        if X.shape[1] != self.x_weights_.shape[0]:
            raise ValueError("X has a different number of features than during fit.")
        Tx = ((X - self.x_mean_) / self.x_std_) @ self.x_weights_
        if Y is None:
            return Tx
        Y = _as_matrix(Y, "Y")
        if Y.shape[1] != self.y_weights_.shape[0]:
            raise ValueError("Y has a different number of features than during fit.")
        return Tx, ((Y - self.y_mean_) / self.y_std_) @ self.y_weights_

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X, Y)

    def score(self, X, Y) -> np.ndarray:
        """Correlation of each canonical pair on (possibly held-out) data."""
        Tx, Ty = self.transform(X, Y)
        return np.array([np.corrcoef(Tx[:, j], Ty[:, j])[0, 1] for j in range(Tx.shape[1])])


def _corr_cols(M: np.ndarray, T: np.ndarray) -> np.ndarray:
    Mc = M - M.mean(axis=0)
    Tc = T - T.mean(axis=0)
    num = Mc.T @ Tc
    den = np.outer(np.linalg.norm(Mc, axis=0), np.linalg.norm(Tc, axis=0))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def canonical_correlations(X, Y, reg: float = 0.0) -> np.ndarray:
    """All ``min(p, q)`` canonical correlations of ``X`` and ``Y``, in descending order."""
    X, Y = _check_pair(X, Y)
    k = min(X.shape[1], Y.shape[1])
    return CCA(n_components=k, reg=reg).fit(X, Y).correlations_


class PLSSVD:
    """Partial least squares SVD: singular vectors of the cross-covariance ``X'Y``.

    ``x_weights_[:, j]`` and ``y_weights_[:, j]`` are unit vectors that maximise
    ``cov(Xa, Yb)`` subject to orthogonality with the earlier pairs.
    ``covariances_`` are the corresponding singular values divided by ``n - 1``.
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, Y):
        k = _check_int(self.n_components, "n_components", 1)
        X, Y = _check_pair(X, Y)
        if k > min(X.shape[1], Y.shape[1]):
            raise ValueError("n_components must be <= min(n_features_X, n_features_Y).")
        Xc, self.x_mean_, self.x_std_ = _center_scale(X, self.scale)
        Yc, self.y_mean_, self.y_std_ = _center_scale(Y, self.scale)
        U, s, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
        A, B = _flip_pair(U[:, :k], Vt[:k].T)
        self.x_weights_, self.y_weights_ = A, B
        self.covariances_ = s[:k] / (X.shape[0] - 1)
        return self

    def transform(self, X, Y=None):
        if not hasattr(self, "x_weights_"):
            raise RuntimeError("PLSSVD instance is not fitted yet; call fit first.")
        X = _as_matrix(X, "X")
        Tx = ((X - self.x_mean_) / self.x_std_) @ self.x_weights_
        if Y is None:
            return Tx
        Y = _as_matrix(Y, "Y")
        return Tx, ((Y - self.y_mean_) / self.y_std_) @ self.y_weights_

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X, Y)


class PLSRegression:
    """PLS2 regression (NIPALS regression mode), using exact SVD weight vectors.

    Attributes
    ----------
    x_weights_ (W), x_loadings_ (P), y_loadings_ (Q), x_scores_ (T),
    x_rotations_ (``W (P'W)^{-1}``, so ``T = X_c R``), coef_ (p x q, in the
    original units), intercept_ (q,).
    """

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, Y):
        k = _check_int(self.n_components, "n_components", 1)
        self._y_1d = np.ndim(Y) == 1
        X, Y = _check_pair(X, Y)
        n, p = X.shape
        q = Y.shape[1]
        if k > min(n - 1, p):
            raise ValueError("n_components must be <= min(n_samples - 1, n_features).")
        Xk, self.x_mean_, self.x_std_ = _center_scale(X, self.scale)
        Yk, self.y_mean_, self.y_std_ = _center_scale(Y, self.scale)
        W = np.zeros((p, k))
        P = np.zeros((p, k))
        Q = np.zeros((q, k))
        T = np.zeros((n, k))
        for j in range(k):
            C = Xk.T @ Yk
            if not np.any(C):
                raise ValueError(f"X and Y have no remaining covariance after {j} components.")
            U, _, _ = np.linalg.svd(C, full_matrices=False)
            w = U[:, 0]
            w = w * (np.sign(w[np.argmax(np.abs(w))]) or 1.0)
            t = Xk @ w
            tt = float(t @ t)
            p_load = Xk.T @ t / tt
            q_load = Yk.T @ t / tt
            Xk = Xk - np.outer(t, p_load)
            Yk = Yk - np.outer(t, q_load)
            W[:, j], P[:, j], Q[:, j], T[:, j] = w, p_load, q_load, t
        self.x_weights_, self.x_loadings_, self.y_loadings_, self.x_scores_ = W, P, Q, T
        self.x_rotations_ = W @ np.linalg.pinv(P.T @ W)
        coef_std = self.x_rotations_ @ Q.T  # standardised units
        self.coef_ = coef_std / self.x_std_[:, None] * self.y_std_[None, :]
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("PLSRegression instance is not fitted yet; call fit first.")

    def transform(self, X) -> np.ndarray:
        """Latent X scores ``X_c R``."""
        self._check_fitted()
        X = _as_matrix(X, "X")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has a different number of features than during fit.")
        return ((X - self.x_mean_) / self.x_std_) @ self.x_rotations_

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        X = _as_matrix(X, "X")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has a different number of features than during fit.")
        pred = X @ self.coef_ + self.intercept_
        return pred[:, 0] if self._y_1d else pred

    def score(self, X, Y) -> float:
        """Uniform-average R^2, as in scikit-learn."""
        Y = _as_matrix(Y, "Y")
        pred = self.predict(X).reshape(Y.shape)
        ss_res = ((Y - pred) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.zeros_like(ss_res), where=ss_tot > 0)
        return float(np.mean(r2))
