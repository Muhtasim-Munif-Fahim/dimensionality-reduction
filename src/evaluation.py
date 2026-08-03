"""Evaluation of dimensionality reduction embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def reconstruction_error(original: pd.DataFrame, reduced: pd.DataFrame, method: str = "pca") -> float:
    if method in ("pca", "svd", "autoencoder"):
        # approximate via projection back to original space
        X = original.to_numpy()
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n = reduced.shape[1]
        approx = U[:, :n] @ np.diag(S[:n]) @ Vt[:n, :] + X.mean(axis=0)
        return float(np.mean((X - approx) ** 2))
    return float("nan")


def silhouette_of_embedding(embedding: pd.DataFrame, labels: pd.Series) -> float:
    scaled = StandardScaler().fit_transform(embedding)
    n_unique = labels.nunique()
    if n_unique < 2 or n_unique >= len(labels):
        return -1.0
    return float(silhouette_score(scaled, labels))


def cluster_separation(embedding: pd.DataFrame, labels: pd.Series) -> float:
    """Mean distance between cluster centroids in embedding space."""
    centroids = embedding.groupby(labels).mean()
    if len(centroids) < 2:
        return 0.0
    from itertools import combinations
    distances = []
    for c1, c2 in combinations(centroids.index, 2):
        d = np.linalg.norm(centroids.loc[c1] - centroids.loc[c2])
        distances.append(d)
    return float(np.mean(distances))
