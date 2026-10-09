"""Local Tangent Space Alignment in NumPy (Zhang & Zha, 2004).

LLE preserves local reconstruction weights; Isomap preserves geodesic
distances. LTSA instead estimates a local tangent basis at each point
(PCA on the neighbourhood) and aligns those tangent coordinates into a
global embedding by minimising the alignment residual. This module is
sklearn-free: k-nearest neighbours, local SVD, and the dense eigenproblem
on the alignment matrix are all implemented with NumPy.

``fit_transform`` returns the embedding. Classic LTSA is a transductive
map; there is no out-of-sample ``transform``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LTSA", "ltsa_reduce"]


class LTSA:
    """Local Tangent Space Alignment via neighbourhood PCA + global align.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3). Also the local tangent
        dimension used in each neighbourhood SVD.
    n_neighbors :
        Number of nearest neighbours in each local patch (including the
        point itself in the classic formulation we use *k* exclusive
        neighbours, then form the patch of size ``k + 1``). Must be at
        least ``n_components`` and strictly less than ``n_samples``.
    reg :
        Floor added to the diagonal of the alignment matrix for numerical
        stability when assembling overlapping patches.
    random_state :
        Unused (kept for API symmetry with t-SNE / NMF). Deterministic.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    alignment_matrix_ :
        Symmetric alignment matrix ``B`` of shape ``(n_samples, n_samples)``.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-9,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.n_neighbors = _validate_n_neighbors(n_neighbors)
        self.reg = _validate_reg(reg)
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.alignment_matrix_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> "LTSA":
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        k_comp = int(self.n_components)
        k_nn = int(self.n_neighbors)

        if k_comp >= n_samples:
            raise ValueError(
                f"n_components={k_comp} must be less than n_samples={n_samples}."
            )
        if k_nn >= n_samples:
            raise ValueError(
                f"n_neighbors={k_nn} must be less than n_samples={n_samples}."
            )
        if k_nn < k_comp:
            raise ValueError(
                f"n_neighbors={k_nn} must be >= n_components={k_comp} "
                "so each local patch has a full tangent basis."
            )
        if k_comp + 1 >= n_samples:
            raise ValueError(
                f"n_components={k_comp} leaves no room for the trivial "
                f"eigenvector (need n_components + 1 < n_samples={n_samples})."
            )

        knn = _knn_indices(matrix, k_nn)
        B = _alignment_matrix(matrix, knn, k_comp, self.reg)
        embedding = _embed_from_alignment(B, k_comp)

        self.embedding_ = np.ascontiguousarray(embedding)
        self.alignment_matrix_ = B
        self.n_components_ = k_comp
        self.n_features_in_ = n_features
        return self.embedding_


def ltsa_reduce(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    reg: float = 1e-9,
    random_state: int | None = None,
):
    """Fit LTSA and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    model = LTSA(
        n_components=n_components,
        n_neighbors=n_neighbors,
        reg=reg,
        random_state=random_state,
    )
    emb = model.fit_transform(X)
    columns = [f"LTSA{i + 1}" for i in range(model.n_components_)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(emb, index=X.index, columns=columns)
    return pd.DataFrame(emb, columns=columns)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_n_neighbors(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    return int(value)


def _validate_reg(value: object) -> float:
    try:
        reg = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("reg must be a non-negative float.") from exc
    if not np.isfinite(reg) or reg < 0.0:
        raise ValueError("reg must be a non-negative float.")
    return reg


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


def _pairwise_squared_euclidean(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _knn_indices(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    d2 = _pairwise_squared_euclidean(X)
    order = np.argsort(d2, axis=1)
    return order[:, 1 : n_neighbors + 1]


def _alignment_matrix(
    X: np.ndarray, knn: np.ndarray, n_components: int, reg: float
) -> np.ndarray:
    """Assemble the LTSA alignment matrix B from local tangent bases."""
    n_samples = X.shape[0]
    B = np.zeros((n_samples, n_samples), dtype=float)
    eye_d = np.eye(n_components)

    for i in range(n_samples):
        # Patch = point i + its k neighbours (size m = k + 1).
        nbrs = np.concatenate(([i], knn[i]))
        # Unique while preserving order (self could appear if graph oddity).
        seen = set()
        patch = []
        for idx in nbrs:
            if idx not in seen:
                seen.add(idx)
                patch.append(int(idx))
        patch_idx = np.asarray(patch, dtype=int)
        m = len(patch_idx)
        if m <= n_components:
            # Degenerate patch: skip contribution.
            continue

        Xi = X[patch_idx]
        Xi_c = Xi - Xi.mean(axis=0, keepdims=True)
        # Economy SVD; leading n_components right/left vectors span the
        # tangent space. We need the left singular vectors (scores).
        try:
            U, S, _Vt = np.linalg.svd(Xi_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        rank = min(n_components, U.shape[1])
        if rank < 1:
            continue
        Theta = U[:, :rank]  # (m, d)
        # Gi = [1/sqrt(m) * 1, Theta]; Wi = I - Gi Gi^T
        ones = np.ones((m, 1), dtype=float) / np.sqrt(m)
        Gi = np.hstack([ones, Theta])
        # Wi = I_m - Gi Gi^T  (orthogonal projector onto complement)
        Wi = np.eye(m) - Gi @ Gi.T
        # B[patch, patch] += Wi^T Wi = Wi (symmetric idempotent)
        B[np.ix_(patch_idx, patch_idx)] += Wi

    # Symmetrise and regularise.
    B = 0.5 * (B + B.T)
    if reg > 0.0:
        B = B + reg * np.eye(n_samples)
    return B


def _embed_from_alignment(B: np.ndarray, n_components: int) -> np.ndarray:
    """Bottom eigenvectors of B, skipping the constant mode."""
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)
    chosen = idx[1 : n_components + 1]
    embedding = eigenvectors[:, chosen]
    for j in range(embedding.shape[1]):
        col = embedding[:, j]
        nz = np.flatnonzero(np.abs(col) > 1e-12)
        if len(nz) and col[nz[0]] < 0.0:
            embedding[:, j] = -col
    return embedding
