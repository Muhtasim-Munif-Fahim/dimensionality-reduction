"""Tests for Gaussian / sparse random projections and the JL bound."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    distance_distortion,
    johnson_lindenstrauss_min_dim,
)


def test_jl_min_dim_formula_and_broadcast():
    expected = math.ceil(4 * math.log(1000) / (0.1**2 / 2 - 0.1**3 / 3))
    assert johnson_lindenstrauss_min_dim(1000, eps=0.1) == expected == 5921
    assert johnson_lindenstrauss_min_dim(1_000_000, eps=0.5) == 664
    arr = johnson_lindenstrauss_min_dim([100, 1000], eps=0.3)
    assert arr.shape == (2,) and arr[0] < arr[1]
    assert johnson_lindenstrauss_min_dim(100, eps=[0.1, 0.5])[0] > johnson_lindenstrauss_min_dim(100, eps=0.5)
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(100, eps=1.0)
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(0)


def test_gaussian_shapes_and_scaling():
    rp = GaussianRandomProjection(n_components=400, random_state=0)
    X = np.random.default_rng(123).normal(size=(50, 2000))
    Y = rp.fit_transform(X)
    assert Y.shape == (50, 400)
    assert rp.components_.shape == (400, 2000)
    assert np.var(rp.components_) == pytest.approx(1 / 400, rel=0.02)
    # norms preserved on average
    ratio = np.sum(Y**2, axis=1) / np.sum(X**2, axis=1)
    assert ratio.mean() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("cls", [GaussianRandomProjection, SparseRandomProjection])
def test_auto_dimension_respects_jl_guarantee(cls):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6000))
    rp = cls(eps=0.3, random_state=0)
    Y = rp.fit_transform(X)
    assert rp.n_components_ == johnson_lindenstrauss_min_dim(40, eps=0.3)
    dist = distance_distortion(X, Y)
    assert dist["max_abs_distortion"] < 0.3
    assert dist["mean"] == pytest.approx(1.0, abs=0.05)


def test_auto_raises_when_target_exceeds_features():
    with pytest.raises(ValueError, match="raise eps"):
        GaussianRandomProjection(eps=0.1).fit(np.zeros((100, 50)))


def test_sparse_density_and_values():
    d, k = 900, 300
    rp = SparseRandomProjection(n_components=k, random_state=3).fit(np.zeros((5, d)))
    R = rp.components_
    assert sparse.isspmatrix_csr(R) or sparse.issparse(R)
    assert rp.density_ == pytest.approx(1 / 30)
    assert R.nnz / (k * d) == pytest.approx(1 / 30, rel=0.08)
    vals = np.unique(np.abs(R.data))
    assert np.allclose(vals, math.sqrt(1 / (rp.density_ * k)))
    assert abs(np.mean(np.sign(R.data))) < 0.05
    ach = SparseRandomProjection(n_components=k, density=1 / 3, random_state=3).fit(np.zeros((5, d)))
    assert ach.components_.nnz / (k * d) == pytest.approx(1 / 3, rel=0.03)


def test_sparse_input_and_sparse_output():
    rng = np.random.default_rng(4)
    dense = rng.poisson(0.05, size=(30, 3000)).astype(float)
    Xs = sparse.csr_matrix(dense)
    rp = SparseRandomProjection(n_components=200, random_state=0)
    Yd = rp.fit_transform(Xs)
    assert isinstance(Yd, np.ndarray) and Yd.shape == (30, 200)
    assert np.allclose(Yd, rp.transform(dense))
    rp2 = SparseRandomProjection(n_components=200, dense_output=False, random_state=0)
    Ys = rp2.fit_transform(Xs)
    assert sparse.issparse(Ys)
    assert np.allclose(Ys.toarray(), Yd)


def test_reproducible_and_validation():
    X = np.random.default_rng(5).normal(size=(10, 100))
    a = GaussianRandomProjection(20, random_state=7).fit_transform(X)
    b = GaussianRandomProjection(20, random_state=7).fit_transform(X)
    assert np.array_equal(a, b)
    s1 = SparseRandomProjection(20, random_state=7).fit_transform(X)
    s2 = SparseRandomProjection(20, random_state=7).fit_transform(X)
    assert np.array_equal(s1, s2)
    with pytest.raises(RuntimeError):
        GaussianRandomProjection(5).transform(X)
    with pytest.raises(ValueError):
        GaussianRandomProjection(0).fit(X)
    with pytest.raises(ValueError):
        GaussianRandomProjection("big").fit(X)
    with pytest.raises(ValueError):
        GaussianRandomProjection(5, eps=0.0)
    with pytest.raises(ValueError):
        SparseRandomProjection(5, density=0.0)
    rp = GaussianRandomProjection(5, random_state=0).fit(X)
    with pytest.raises(ValueError):
        rp.transform(X[:, :50])
    with pytest.raises(ValueError):
        rp.fit(np.array([[np.nan, 1.0]]))


def test_inverse_transform_recovers_row_space():
    rng = np.random.default_rng(6)
    rp = GaussianRandomProjection(30, random_state=0).fit(np.zeros((2, 60)))
    # signal living in the projection's row space is recovered exactly
    coeffs = rng.normal(size=(8, 30))
    X = coeffs @ rp.components_
    back = rp.inverse_transform(rp.transform(X))
    assert np.allclose(back, X, atol=1e-8)
    srp = SparseRandomProjection(30, density=0.5, random_state=0).fit(np.zeros((2, 60)))
    X2 = coeffs @ srp.components_.toarray()
    assert np.allclose(srp.inverse_transform(srp.transform(X2)), X2, atol=1e-8)
    with pytest.raises(ValueError):
        rp.inverse_transform(np.zeros((3, 4)))


def test_distance_distortion_sampling_and_identity():
    X = np.random.default_rng(7).normal(size=(25, 10))
    full = distance_distortion(X, X)
    assert full["max_abs_distortion"] == pytest.approx(0.0)
    assert full["ratios"].size == 25 * 24 // 2
    sampled = distance_distortion(X, 2 * X, n_pairs=100, random_state=0)
    assert sampled["mean"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        distance_distortion(X, X[:5])
