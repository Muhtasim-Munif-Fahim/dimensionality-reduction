"""Tests for autoencoder and evaluation."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoencoder import NumpyAutoencoder
from evaluation import silhouette_of_embedding, reconstruction_error, cluster_separation
from data_generation import generate_high_dim_data


class TestAutoencoder:
    def test_encode_shape(self):
        ae = NumpyAutoencoder(n_input=10, n_hidden=4)
        X = np.random.randn(20, 10)
        h = ae.encode(X)
        assert h.shape == (20, 4)

    def test_reconstruct_error_decreases(self):
        ae = NumpyAutoencoder(n_input=10, n_hidden=5, learning_rate=0.01)
        X = np.random.randn(30, 10)
        loss_before = ae.reconstruct_error(X)
        ae.fit(X, epochs=20)
        loss_after = ae.reconstruct_error(X)
        assert loss_after <= loss_before


class TestEvaluation:
    def test_silhouette_bounds(self):
        df, labels = generate_high_dim_data(n_samples=100, n_features=15)
        from pca import pca_manual
        emb, _ = pca_manual(df, n_components=2)
        score = silhouette_of_embedding(emb, labels)
        assert -1.0 <= score <= 1.0

    def test_reconstruction_error_nonneg(self):
        df, _ = generate_high_dim_data(n_samples=50, n_features=10)
        from pca import pca_svd
        emb, _ = pca_svd(df, n_components=2)
        err = reconstruction_error(df, emb)
        assert err >= 0

    def test_separation_positive(self):
        df, labels = generate_high_dim_data(n_samples=100, n_features=15)
        from pca import pca_manual
        emb, _ = pca_manual(df, n_components=2)
        assert cluster_separation(emb, labels) > 0
