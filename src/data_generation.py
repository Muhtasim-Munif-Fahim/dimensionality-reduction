"""Synthetic high-dimensional data generation with cluster structure."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_high_dim_data(
    n_samples: int = 500,
    n_features: int = 50,
    n_clusters: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = []
    labels = []
    # Create a low-dimensional latent structure embedded in high-dim space
    latent_dim = min(n_clusters, 5)
    cluster_centers = rng.normal(0, 4, size=(n_clusters, latent_dim))

    for i in range(n_samples):
        cluster = i % n_clusters
        latent = cluster_centers[cluster] + rng.normal(0, 0.5, latent_dim)
        # Project latent into high-dim via random projection
        projection = rng.normal(0, 1, size=(latent_dim, n_features)) if i == 0 else projection
        sample = latent @ projection + rng.normal(0, 0.1, n_features)
        data.append(sample)
        labels.append(cluster)

    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(n_features)])
    return df, pd.Series(labels, name="cluster")


def generate_cluster_labels(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.Series:
    rng = np.random.default_rng(random_state)
    return pd.Series(rng.integers(0, n_clusters, len(df)), name="cluster")
