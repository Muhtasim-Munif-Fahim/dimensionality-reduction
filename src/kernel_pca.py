"""Kernel PCA via a centred Gram matrix and eigendecomposition.

Linear PCA recovers directions of variance in the input space. Kernel PCA
(Schölkopf, Smola & Müller, 1998) does the same in a feature space induced
by a positive-definite kernel, using only the Gram matrix of pairwise
kernel evaluations. Supported kernels are RBF, linear, and polynomial.

``fit`` stores the training rows and the leading eigenvectors of the
centred Gram matrix. ``transform`` maps new rows by the Nyström-style
projection onto those eigenvectors (out-of-sample extension).
"""

from __future__ import annotations

import numpy as np

__all__ = ["KernelPCA", "kernel_pca_reduce"]


class KernelPCA:
    """Kernel principal component analysis.

    Parameters
    ----------
    n_components :
        Embedding dimension.
    kernel :
        ``"rbf"`` (default), ``"linear"``, or ``"poly"``.
    gamma :
        Kernel coefficient for RBF / poly. ``None`` uses
        ``1 / (n_features * Var(X))`` (or 1 when that variance is 0).
    degree :
        Polynomial degree (``kernel="poly"`` only).
    coef0 :
        Independent term in the polynomial kernel.
    random_state / seed :
        Unused (kept for API symmetry). The eigendecomposition is
        deterministic up to eigenvector sign, which is fixed below.

    Attributes
    ----------
    embedding_ :
        Training embedding, shape ``(n_samples, n_components_)``.
    eigenvalues_ :
        Leading eigenvalues of the centred Gram matrix.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    X_fit_ :
        Training matrix used to build the Gram matrix.
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        random_state: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.kernel = _validate_kernel(kernel)
        self.gamma = _validate_gamma(gamma)
        self.degree = _validate_degree(degree)
        self.coef0 = _validate_coef0(coef0)
        # Prefer random_state; seed is an alias used by some callers.
        rs = random_state if random_state is not None else seed
        self.random_state = _validate_random_state(rs)
        self.seed = self.random_state
        self.embedding_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.X_fit_: np.ndarray | None = None
        self._gamma: float | None = None
        self._K_fit_row_means: np.ndarray | None = None
        self._K_fit_mean: float | None = None

    def fit(self, X) -> KernelPCA:
        """Fit Kernel PCA on ``X`` and store the training embedding."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Fit on ``X`` and return the low-dimensional embedding."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")
        k = int(self.n_components)
        if k >= n_samples:
            raise ValueError(
                f"n_components={k} must be less than n_samples={n_samples}."
            )

        if self.gamma is None:
            gamma = _scale_gamma(matrix)
        else:
            gamma = float(self.gamma)

        K = _kernel_matrix(
            matrix, matrix, self.kernel, gamma, self.degree, self.coef0
        )
        K_c, row_means, k_mean = _center_gram(K)
        eigenvalues, eigenvectors = np.linalg.eigh(K_c)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep positive eigenvalues only for the embedding scale.
        positives = np.maximum(eigenvalues[:k], 0.0)
        # Alpha = V / sqrt(lambda); embedding = K_c @ alpha = V * sqrt(lambda)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(positives > 1e-12, 1.0 / np.sqrt(positives), 0.0)
        alphas = eigenvectors[:, :k] * scale
        embedding = K_c @ alphas
        # Deterministic sign: flip each column so its first nonzero entry is >= 0.
        for j in range(embedding.shape[1]):
            col = embedding[:, j]
            nz = np.flatnonzero(np.abs(col) > 1e-12)
            if len(nz) and col[nz[0]] < 0.0:
                embedding[:, j] = -col
                alphas[:, j] = -alphas[:, j]

        self.X_fit_ = np.array(matrix, dtype=float, copy=True)
        self._gamma = gamma
        self._K_fit_row_means = row_means
        self._K_fit_mean = k_mean
        self.eigenvalues_ = positives
        self.eigenvectors_ = alphas
        self.embedding_ = np.ascontiguousarray(embedding)
        self.n_components_ = k
        self.n_features_in_ = n_features
        return self.embedding_

    def transform(self, X) -> np.ndarray:
        """Project new rows into the fitted kernel PCA space."""
        if (
            self.X_fit_ is None
            or self.eigenvectors_ is None
            or self._K_fit_row_means is None
            or self._K_fit_mean is None
            or self._gamma is None
        ):
            raise ValueError("KernelPCA must be fitted before transform")
        matrix = _as_dense(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but KernelPCA was fitted "
                f"on {self.n_features_in_}"
            )
        if matrix.shape[0] == 0:
            return np.empty((0, int(self.n_components_)))
        K = _kernel_matrix(
            matrix,
            self.X_fit_,
            self.kernel,
            self._gamma,
            self.degree,
            self.coef0,
        )
        K_c = _center_gram_out_of_sample(
            K, self._K_fit_row_means, self._K_fit_mean
        )
        return K_c @ self.eigenvectors_


def kernel_pca_reduce(
    X,
    n_components: int = 2,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
    random_state: int | None = None,
):
    """Fit Kernel PCA and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"KPCA{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_kernel(value: object) -> str:
    if value not in ("rbf", "linear", "poly"):
        raise ValueError('kernel must be "rbf", "linear", or "poly".')
    return str(value)


def _validate_gamma(value: object) -> float | None:
    if value is None:
        return None
    gamma = float(value)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite number, or None.")
    return gamma


def _validate_degree(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("degree must be a positive integer.")
    return int(value)


def _validate_coef0(value: object) -> float:
    coef0 = float(value)
    if not np.isfinite(coef0):
        raise ValueError("coef0 must be finite.")
    return coef0


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


def _scale_gamma(X: np.ndarray) -> float:
    if X.size == 0 or X.shape[1] == 0:
        return 1.0
    var = float(np.var(X))
    if not np.isfinite(var) or var <= 0.0:
        return 1.0
    return 1.0 / (X.shape[1] * var)


def _kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    if kernel == "linear":
        return X @ Y.T
    dots = X @ Y.T
    if kernel == "poly":
        return (gamma * dots + coef0) ** degree
    x2 = np.einsum("ij,ij->i", X, X)
    y2 = np.einsum("ij,ij->i", Y, Y)
    d2 = x2[:, None] + y2[None, :] - 2.0 * dots
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _center_gram(K: np.ndarray):
    """Centre a square Gram matrix; return (K_c, row_means, grand_mean)."""
    row_means = K.mean(axis=1)
    k_mean = float(K.mean())
    K_c = K - row_means[:, None] - row_means[None, :] + k_mean
    K_c = 0.5 * (K_c + K_c.T)
    return K_c, row_means, k_mean


def _center_gram_out_of_sample(
    K: np.ndarray, train_row_means: np.ndarray, train_mean: float
) -> np.ndarray:
    """Centre a rectangular Gram matrix K(X_new, X_train)."""
    row_means = K.mean(axis=1)
    return K - train_row_means[None, :] - row_means[:, None] + train_mean
