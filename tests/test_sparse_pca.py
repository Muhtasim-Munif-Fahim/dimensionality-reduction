"""Tests for Sparse PCA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparse_pca import SparsePCA, soft_threshold, sparse_pca_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _block_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    X = np.column_stack(
        [
            z1 + 0.05 * rng.normal(size=n),
            z1 + 0.05 * rng.normal(size=n),
            z1 + 0.05 * rng.normal(size=n),
            z2 + 0.05 * rng.normal(size=n),
            z2 + 0.05 * rng.normal(size=n),
            z2 + 0.05 * rng.normal(size=n),
            0.05 * rng.normal(size=n),
            0.05 * rng.normal(size=n),
        ]
    )
    return X


def test_soft_threshold_known_values():
    out = soft_threshold([-2.0, -0.5, 0.0, 0.5, 2.0], 1.0)
    assert np.allclose(out, [-1.0, 0.0, 0.0, 0.0, 1.0])


def test_fit_transform_shape_and_attributes():
    X = _block_data()
    model = SparsePCA(n_components=2, alpha=0.05).fit(X)
    emb = model.transform(X)
    assert emb.shape == (200, 2)
    assert model.components_.shape == (2, 8)
    assert model.mean_.shape == (8,)
    assert np.all(np.isfinite(emb))


def test_sparsity_increases_with_alpha():
    X = _block_data(seed=1)
    dense = SparsePCA(n_components=1, alpha=0.0).fit(X).components_[0]
    sparse = SparsePCA(n_components=1, alpha=0.2).fit(X).components_[0]
    assert np.count_nonzero(np.abs(sparse) > 1e-12) <= np.count_nonzero(np.abs(dense) > 1e-12)


def test_components_are_unit_norm():
    X = _block_data(seed=2)
    comps = SparsePCA(n_components=2, alpha=0.05).fit(X).components_
    norms = np.linalg.norm(comps, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_dataframe_reduce():
    X = pd.DataFrame(_block_data(n=50), columns=[f"f{i}" for i in range(8)])
    emb = sparse_pca_reduce(X, n_components=2, alpha=0.05)
    assert list(emb.columns) == ["SPC1", "SPC2"]
    assert emb.shape == (50, 2)


def test_rejects_bad_params():
    X = _block_data(n=30)
    _raises(ValueError, lambda: SparsePCA(n_components=0))
    _raises(ValueError, lambda: SparsePCA(alpha=-0.1))
    _raises(ValueError, lambda: SparsePCA(n_components=20).fit(X))


def test_transform_before_fit_raises():
    _raises(ValueError, lambda: SparsePCA().transform(_block_data(n=20)))


def test_deterministic():
    X = _block_data(seed=3)
    a = SparsePCA(n_components=2, alpha=0.05).fit_transform(X)
    b = SparsePCA(n_components=2, alpha=0.05).fit_transform(X)
    assert np.allclose(a, b)
