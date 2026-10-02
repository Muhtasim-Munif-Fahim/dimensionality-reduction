"""Tests for Factor Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factor_analysis import FactorAnalysis, factor_analysis_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _two_factor_data(n: int = 200, seed: int = 0):
    """Data generated from a known 2-factor FA model."""
    rng = np.random.default_rng(seed)
    n_features = 6
    W = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.0],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.0, 0.7],
        ],
        dtype=float,
    )
    Z = rng.normal(size=(n, 2))
    psi = np.array([0.2, 0.3, 0.25, 0.2, 0.3, 0.25])
    noise = rng.normal(size=(n, n_features)) * np.sqrt(psi)
    X = Z @ W.T + noise
    return X, W, psi


class TestFactorAnalysisBasic:
    def test_fit_transform_shape(self):
        X, _, _ = _two_factor_data()
        emb = FactorAnalysis(n_components=2, max_iter=200, tol=1e-4).fit_transform(X)
        assert emb.shape == (200, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_attributes(self):
        X, _, _ = _two_factor_data(n=120)
        model = FactorAnalysis(n_components=2, max_iter=200, tol=1e-4).fit(X)
        assert model.components_ is not None
        assert model.components_.shape == (2, 6)
        assert model.noise_variance_ is not None
        assert model.noise_variance_.shape == (6,)
        assert np.all(model.noise_variance_ > 0)
        assert model.mean_ is not None
        assert model.mean_.shape == (6,)
        assert model.n_iter_ is not None and model.n_iter_ >= 1
        assert model.loglike_ is not None and np.isfinite(model.loglike_)

    def test_transform_matches_fit_transform(self):
        X, _, _ = _two_factor_data(n=80, seed=1)
        model = FactorAnalysis(n_components=2, max_iter=150, tol=1e-4).fit(X)
        a = model.transform(X)
        b = FactorAnalysis(n_components=2, max_iter=150, tol=1e-4).fit_transform(X)
        assert np.allclose(a, b, atol=1e-8)

    def test_deterministic(self):
        X, _, _ = _two_factor_data(n=100, seed=2)
        a = FactorAnalysis(n_components=2, max_iter=100, tol=1e-3).fit_transform(X)
        b = FactorAnalysis(n_components=2, max_iter=100, tol=1e-3).fit_transform(X)
        assert np.allclose(a, b)

    def test_recovers_correlated_structure(self):
        # Two latent blocks: first 3 features share factor 1, last 3 share factor 2.
        rng = np.random.default_rng(3)
        n = 300
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        X = np.column_stack(
            [
                z1 + 0.1 * rng.normal(size=n),
                z1 + 0.1 * rng.normal(size=n),
                z1 + 0.1 * rng.normal(size=n),
                z2 + 0.1 * rng.normal(size=n),
                z2 + 0.1 * rng.normal(size=n),
                z2 + 0.1 * rng.normal(size=n),
            ]
        )
        model = FactorAnalysis(n_components=2, max_iter=300, tol=1e-4).fit(X)
        # Absolute loadings: each feature should load strongly on one factor.
        abs_load = np.abs(model.components_)
        # For each feature, max loading across factors should dominate.
        assert abs_load.max(axis=0).mean() > 0.3

    def test_helper_reduce(self):
        X, _, _ = _two_factor_data(n=60, seed=4)
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out = factor_analysis_reduce(frame, n_components=2, max_iter=100, tol=1e-3)
        assert list(out.columns) == ["FA1", "FA2"]
        assert len(out) == 60

    def test_transform_before_fit_raises(self):
        _raises(
            RuntimeError,
            lambda: FactorAnalysis(n_components=2).transform(np.zeros((5, 3))),
        )

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: FactorAnalysis(n_components=0))
        _raises(ValueError, lambda: FactorAnalysis(max_iter=0))
        _raises(ValueError, lambda: FactorAnalysis(tol=-1.0))

    def test_n_components_exceeds_features_raises(self):
        X = np.random.default_rng(0).normal(size=(30, 3))
        _raises(ValueError, lambda: FactorAnalysis(n_components=4).fit(X))

    def test_feature_mismatch_raises(self):
        X = np.random.default_rng(0).normal(size=(40, 4))
        model = FactorAnalysis(n_components=2, max_iter=50).fit(X)
        _raises(ValueError, lambda: model.transform(np.zeros((5, 3))))

    def test_single_sample_raises(self):
        _raises(
            ValueError,
            lambda: FactorAnalysis(n_components=1).fit(np.zeros((1, 3))),
        )
