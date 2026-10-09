"""Tests for Local Tangent Space Alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ltsa import LTSA, ltsa_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _two_blobs(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-3.0, scale=0.3, size=(n_per, 5))
    b = rng.normal(loc=3.0, scale=0.3, size=(n_per, 5))
    X = np.vstack([a, b])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


def _s_curve_ish(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.5 * np.pi, 1.5 * np.pi, n)
    X = np.column_stack([np.sin(t), 2.0 * t / np.pi, np.sign(t) * (np.cos(t) - 1.0)])
    X = X + 0.02 * rng.normal(size=X.shape)
    return X, t


def test_shape_and_finite():
    X, _ = _two_blobs()
    emb = LTSA(n_components=2, n_neighbors=8).fit_transform(X)
    assert emb.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(emb))


def test_attributes():
    X, _ = _two_blobs(n_per=30)
    model = LTSA(n_components=2, n_neighbors=8).fit(X)
    assert model.embedding_.shape == (60, 2)
    assert model.alignment_matrix_.shape == (60, 60)
    assert model.n_components_ == 2
    assert model.n_features_in_ == 5


def test_deterministic():
    X, _ = _two_blobs()
    a = LTSA(n_components=2, n_neighbors=8).fit_transform(X)
    b = LTSA(n_components=2, n_neighbors=8).fit_transform(X)
    assert np.allclose(a, b)


def test_separates_blobs():
    X, y = _two_blobs(n_per=50, seed=1)
    emb = LTSA(n_components=2, n_neighbors=10).fit_transform(X)
    # Mean of each blob should be far apart relative to within-blob spread.
    c0 = emb[y == 0].mean(axis=0)
    c1 = emb[y == 1].mean(axis=0)
    within = 0.5 * (
        emb[y == 0].std(axis=0).mean() + emb[y == 1].std(axis=0).mean()
    )
    sep = np.linalg.norm(c0 - c1)
    assert sep > 2.0 * within


def test_s_curve_embedding_finite():
    X, _ = _s_curve_ish()
    emb = LTSA(n_components=2, n_neighbors=12).fit_transform(X)
    assert emb.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(emb))


def test_dataframe_helper():
    X, _ = _two_blobs(n_per=25)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    out = ltsa_reduce(df, n_components=2, n_neighbors=8)
    assert list(out.columns) == ["LTSA1", "LTSA2"]
    assert out.shape == (50, 2)
    assert list(out.index) == list(df.index)


def test_invalid_n_components():
    X, _ = _two_blobs(n_per=20)
    _raises(ValueError, lambda: LTSA(n_components=0).fit_transform(X))
    _raises(ValueError, lambda: LTSA(n_components=X.shape[0]).fit_transform(X))


def test_neighbors_must_cover_tangent():
    X, _ = _two_blobs(n_per=20)
    _raises(
        ValueError,
        lambda: LTSA(n_components=3, n_neighbors=2).fit_transform(X),
    )


def test_invalid_n_neighbors():
    X, _ = _two_blobs(n_per=15)
    _raises(ValueError, lambda: LTSA(n_neighbors=0).fit_transform(X))
    _raises(
        ValueError,
        lambda: LTSA(n_neighbors=X.shape[0]).fit_transform(X),
    )
