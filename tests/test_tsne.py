"""Tests for classic exact t-SNE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsne import TSNE, tsne_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _two_blobs(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-3.0, scale=0.4, size=(n_per, 6))
    b = rng.normal(loc=3.0, scale=0.4, size=(n_per, 6))
    X = np.vstack([a, b])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


class TestTSNEBasic:
    def test_fit_transform_shape(self):
        X, _ = _two_blobs()
        emb = TSNE(n_components=2, perplexity=10.0, n_iter=250, random_state=0).fit_transform(X)
        assert emb.shape == (80, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_embedding(self):
        X, _ = _two_blobs()
        model = TSNE(n_components=2, perplexity=10.0, n_iter=200, random_state=1).fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (80, 2)
        assert model.n_iter_ is not None and model.n_iter_ >= 1
        assert model.kl_divergence_ is not None
        assert np.isfinite(model.kl_divergence_)

    def test_reproducible_with_seed(self):
        X, _ = _two_blobs()
        a = TSNE(n_components=2, perplexity=10.0, n_iter=200, random_state=7).fit_transform(X)
        b = TSNE(n_components=2, perplexity=10.0, n_iter=200, random_state=7).fit_transform(X)
        assert np.allclose(a, b)

    def test_separates_two_blobs(self):
        X, y = _two_blobs(n_per=35, seed=2)
        emb = TSNE(
            n_components=2,
            perplexity=12.0,
            learning_rate=200.0,
            n_iter=400,
            random_state=0,
        ).fit_transform(X)
        # Mean distance between cluster centres should exceed within-cluster spread.
        c0 = emb[y == 0].mean(axis=0)
        c1 = emb[y == 1].mean(axis=0)
        within = 0.5 * (
            np.linalg.norm(emb[y == 0] - c0, axis=1).mean()
            + np.linalg.norm(emb[y == 1] - c1, axis=1).mean()
        )
        between = np.linalg.norm(c0 - c1)
        assert between > within

    def test_dataframe_helper(self):
        X, _ = _two_blobs(n_per=20)
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out = tsne_reduce(frame, n_components=2, perplexity=8.0, random_state=0)
        assert list(out.columns) == ["TSNE1", "TSNE2"]
        assert len(out) == 40

    def test_perplexity_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(10, 3))
        _raises(ValueError, lambda: TSNE(perplexity=10.0).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: TSNE(n_components=0))
        _raises(ValueError, lambda: TSNE(perplexity=-1.0))
        _raises(ValueError, lambda: TSNE(learning_rate=0.0))
        _raises(ValueError, lambda: TSNE(n_iter=0))

    def test_single_sample_raises(self):
        _raises(ValueError, lambda: TSNE(perplexity=1.0).fit_transform(np.zeros((1, 3))))
