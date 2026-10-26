"""Tests for Sliced Inverse Regression and SAVE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sir import SAVE, SlicedInverseRegression, slice_response, subspace_distance  # noqa: E402


def _data(n=2000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng


def test_slice_response_equal_sizes_and_categorical():
    y = np.arange(100.0)
    labels = slice_response(y, 10)
    assert np.all(np.bincount(labels) == 10)
    assert np.all(np.diff(labels[np.argsort(y)]) >= 0)
    cat = slice_response(np.array(["a", "b", "a", "c"]), 5)
    assert sorted(set(cat)) == [0, 1, 2]
    ties = slice_response(np.r_[np.zeros(50), np.arange(50.0)], 10)
    assert len(set(ties[:50])) == 1
    with pytest.raises(ValueError):
        slice_response(np.ones(10), 5)
    with pytest.raises(ValueError):
        slice_response(y, 1)


def test_subspace_distance():
    A = np.eye(4)[:, :2]
    assert subspace_distance(A, A @ np.array([[2.0, 1.0], [0.0, 3.0]])) == pytest.approx(0.0, abs=1e-12)
    assert subspace_distance(A, np.eye(4)[:, 2:]) == pytest.approx(2.0)


def test_sir_recovers_single_index_monotone_link():
    X, rng = _data()
    beta = np.array([1.0, -1.0, 0.5, 0, 0, 0])
    y = (X @ beta) ** 3 + 0.5 * rng.normal(size=X.shape[0])
    sir = SlicedInverseRegression(n_components=1).fit(X, y)
    cos = abs(sir.directions_[:, 0] @ beta) / np.linalg.norm(beta)
    assert cos > 0.98
    assert sir.explained_ratio_[0] > 0.8
    assert sir.estimate_dimension() == 1


def test_sir_two_dimensional_central_subspace_on_correlated_predictors():
    rng = np.random.default_rng(1)
    n, p = 3000, 5
    L = np.tril(rng.normal(size=(p, p))) + 2 * np.eye(p)
    X = rng.normal(size=(n, p)) @ L.T + 3.0
    b1 = np.array([1.0, 0, 0, 0, 0])
    b2 = np.array([0, 1.0, 1.0, 0, 0])
    s1 = X @ b1
    s2 = X @ b2
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = (s2 - s2.mean()) / s2.std()
    # Li (1991) eq. 6.3 rational model on a correlated Gaussian design.
    y = s1 / (0.5 + (s2 + 1.5) ** 2) + 0.1 * rng.normal(size=n)
    sir = SlicedInverseRegression(n_components=2, n_slices=10).fit(X, y)
    assert subspace_distance(sir.directions_, np.c_[b1, b2]) < 0.35
    assert sir.estimate_dimension() >= 1


def test_save_finds_symmetric_dependence_that_sir_misses():
    X, rng = _data(n=3000, seed=2)
    beta = np.array([0, 1.0, 0, 0, 0, 0])
    y = (X @ beta) ** 2 + 0.2 * rng.normal(size=X.shape[0])
    sir = SlicedInverseRegression(n_components=1).fit(X, y)
    save = SAVE(n_components=1).fit(X, y)
    assert abs(save.directions_[:, 0] @ beta) > 0.98
    assert abs(sir.directions_[:, 0] @ beta) < abs(save.directions_[:, 0] @ beta)
    assert sir.eigenvalues_[0] < 0.05  # inverse mean curve is flat


def test_null_dimension_not_rejected_when_y_is_independent():
    X, rng = _data(n=1500, p=4, seed=3)
    y = rng.normal(size=1500)
    sir = SlicedInverseRegression().fit(X, y)
    tests = sir.dimension_test()
    assert tests[0]["d"] == 0 and tests[0]["df"] == 4 * 9
    assert tests[0]["p_value"] > 0.01
    assert sir.estimate_dimension(alpha=0.01) == 0


def test_categorical_response_matches_lda_direction():
    rng = np.random.default_rng(4)
    n = 600
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 3))
    X[:, 0] += 2.5 * y
    sir = SlicedInverseRegression().fit(pd.DataFrame(X), pd.Series(y))
    assert sir.n_slices_ == 2
    assert abs(sir.directions_[0, 0]) > 0.97


def test_transform_invariance_and_validation():
    X, rng = _data(n=800, p=4, seed=5)
    y = X[:, 0] + 0.1 * rng.normal(size=800)
    sir = SlicedInverseRegression(n_components=2)
    T = sir.fit_transform(X, y)
    assert T.shape == (800, 2)
    np.testing.assert_allclose(sir.transform(X), T)
    np.testing.assert_allclose(np.linalg.norm(sir.directions_, axis=0), 1.0)
    # Affine invariance of the estimated subspace.
    A = np.diag([2.0, 0.5, 1.0, 3.0])
    d1 = sir.directions_[:, :1]
    d2 = SlicedInverseRegression().fit(X @ A + 7.0, y).directions_
    assert subspace_distance(np.linalg.solve(A, d1), d2) < 1e-8
    with pytest.raises(ValueError):
        SlicedInverseRegression(n_components=0).fit(X, y)
    with pytest.raises(ValueError):
        SlicedInverseRegression().fit(X, y[:10])
    with pytest.raises(ValueError):
        SlicedInverseRegression().fit(np.c_[X, X[:, 0]], y)
    with pytest.raises(ValueError):
        SAVE(reg=-1).fit(X, y)
    with pytest.raises(RuntimeError):
        SAVE().transform(X)
    with pytest.raises(ValueError):
        sir.transform(X[:, :2])
    ok = SlicedInverseRegression(reg=0.01).fit(np.c_[X, X[:, 0]], y)
    assert ok.directions_.shape == (5, 1)
