"""PCA implementations: manual (numpy eigendecomposition) and SVD-based.

Both helpers center the columns before factoring. Sparse TF-IDF-style inputs
should use ``truncated_svd.TruncatedSVD`` instead: it does not center, so
structural zeros stay zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pca_manual(
    X: pd.DataFrame, n_components: int = 2
) -> tuple[pd.DataFrame, dict]:
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]
    projected = X_centered @ components

    explained = eigenvalues[idx] / eigenvalues.sum()
    result = projected.copy() if isinstance(projected, pd.DataFrame) else pd.DataFrame(projected)
    result.columns = [f"PC{i+1}" for i in range(n_components)]
    return result, {
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance": float(explained.sum()),
    }


def pca_svd(X: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, dict]:
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    projected = U[:, :n_components] * S[:n_components]
    total_var = np.sum(S**2)
    explained = (S[:n_components] ** 2) / total_var
    return pd.DataFrame(projected, columns=[f"SVD{i+1}" for i in range(n_components)]), {
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance": float(explained.sum()),
    }


def variance_retained(X: pd.DataFrame, n_components: int = 2) -> float:
    _, svd_info = pca_svd(X, n_components)
    return float(svd_info["cumulative_variance"])
