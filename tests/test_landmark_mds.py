"""Tests for Landmark MDS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from landmark_mds import LandmarkMDS, landmark_mds_reduce
from mds import ClassicalMDS


def _procrustes_rmse(Y: np.ndarray, X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc, full_matrices=False)
    return float(np.sqrt(np.mean((Yc @ (U @ Vt) - Xc) ** 2)))


def _planar(n=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2)) * [3.0, 1.0]
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    return Z, Z @ Q[:2] + 4.0


@pytest.mark.parametrize("selection", ["maxmin", "random"])
def test_recovers_rank_two_configuration(selection):
    Z, X = _planar()
    emb = LandmarkMDS(2, n_landmarks=10, selection=selection, random_state=1).fit_transform(X)
    assert emb.shape == (200, 2)
    assert _procrustes_rmse(emb, Z) < 1e-8


def test_matches_classical_mds_when_all_points_are_landmarks():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    lmds = LandmarkMDS(3, n_landmarks=40, random_state=0).fit_transform(X)
    cmds = ClassicalMDS(3).fit_transform(X)
    assert _procrustes_rmse(lmds, cmds) < 1e-8


def test_out_of_sample_transform_consistent():
    _, X = _planar(seed=5)
    model = LandmarkMDS(2, n_landmarks=12, random_state=0).fit(X[:150])
    full = LandmarkMDS(2, n_landmarks=12, random_state=0).fit(X[:150])
    np.testing.assert_allclose(model.transform(X[150:]), full.transform(X[150:]))
    # Distances between held-out points are preserved for planar data.
    Y = model.transform(X[150:])
    dX = np.linalg.norm(X[150:, None] - X[None, 150:], axis=-1)
    dY = np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
    np.testing.assert_allclose(dY, dX, atol=1e-7)


def test_maxmin_picks_distinct_spread_landmarks():
    _, X = _planar()
    model = LandmarkMDS(2, n_landmarks=8, random_state=0).fit(X)
    assert len(set(model.landmark_indices_.tolist())) == 8
    assert model.n_components_ == 2
    assert np.all(model.eigenvalues_ > 0)


def test_reduce_returns_dataframe():
    _, X = _planar(n=30)
    df = landmark_mds_reduce(pd.DataFrame(X, index=[f"r{i}" for i in range(30)]), n_landmarks=6)
    assert list(df.columns) == ["LMDS1", "LMDS2"]
    assert df.index[0] == "r0"


def test_validation():
    with pytest.raises(ValueError):
        LandmarkMDS(n_components=0)
    with pytest.raises(ValueError):
        LandmarkMDS(n_components=3, n_landmarks=3)
    with pytest.raises(ValueError):
        LandmarkMDS(selection="kmeans")
    with pytest.raises(RuntimeError):
        LandmarkMDS().transform(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        LandmarkMDS(2, 5).fit(np.array([[0.0, np.nan], [1.0, 2.0], [3.0, 4.0]]))
    model = LandmarkMDS(2, 5).fit(np.random.default_rng(0).normal(size=(20, 3)))
    with pytest.raises(ValueError):
        model.transform(np.zeros((2, 4)))
