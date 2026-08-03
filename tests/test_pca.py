"""Tests for data generation and PCA."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_high_dim_data
from pca import pca_manual, pca_svd, variance_retained


class TestDataGeneration:
    def test_shape(self):
        df, labels = generate_high_dim_data(n_samples=100, n_features=20)
        assert df.shape == (100, 20)
        assert len(labels) == 100

    def test_has_clusters(self):
        df, labels = generate_high_dim_data(n_samples=100, n_features=20, n_clusters=4)
        assert labels.nunique() == 4


class TestPCA:
    def test_manual_pca_shape(self):
        df, _ = generate_high_dim_data(n_samples=100, n_features=20)
        emb, info = pca_manual(df, n_components=2)
        assert emb.shape == (100, 2)
        assert "explained_variance_ratio" in info

    def test_svd_pca_shape(self):
        df, _ = generate_high_dim_data(n_samples=100, n_features=20)
        emb, info = pca_svd(df, n_components=3)
        assert emb.shape == (100, 3)

    def test_variance_retained_bounds(self):
        df, _ = generate_high_dim_data(n_samples=100, n_features=20)
        v = variance_retained(df, n_components=2)
        assert 0.0 <= v <= 1.0
