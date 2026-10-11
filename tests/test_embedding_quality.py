"""Tests for rank-based embedding quality metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedding_quality import (
    continuity,
    coranking_matrix,
    embedding_quality,
    lcmc,
    neighborhood_preservation,
    rnx_auc,
    rnx_curve,
    trustworthiness,
)
from pca import pca_svd


def _data(n=120, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = X[:, :2] + 0.1 * rng.normal(size=(n, 2))
    return X, Y, rng


@pytest.mark.parametrize("k", [1, 5, 10, 30])
def test_trustworthiness_and_continuity_match_sklearn(k):
    sk = pytest.importorskip("sklearn.manifold")
    X, Y, _ = _data()
    assert trustworthiness(X, Y, k) == pytest.approx(sk.trustworthiness(X, Y, n_neighbors=k), abs=1e-12)
    assert continuity(X, Y, k) == pytest.approx(sk.trustworthiness(Y, X, n_neighbors=k), abs=1e-12)


def test_perfect_embedding_scores_one():
    X, _, _ = _data(n=60)
    # Any similarity transform preserves all neighbour ranks.
    rot, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))
    Z = 3.0 * X @ rot + 5.0
    for k in (1, 5, 20):
        assert trustworthiness(X, Z, k) == pytest.approx(1.0)
        assert continuity(X, Z, k) == pytest.approx(1.0)
        assert neighborhood_preservation(X, Z, k) == pytest.approx(1.0)
    assert np.allclose(rnx_curve(X, Z), 1.0)
    assert rnx_auc(X, Z) == pytest.approx(1.0)
    Q = coranking_matrix(X, Z)
    assert np.array_equal(Q, np.diag(np.full(59, 60)))


def test_coranking_matrix_margins():
    X, Y, _ = _data(n=40)
    Q = coranking_matrix(X, Y)
    assert Q.shape == (39, 39)
    # Every point has exactly one neighbour at each rank, in both spaces.
    assert np.all(Q.sum(axis=0) == 40)
    assert np.all(Q.sum(axis=1) == 40)


def test_random_embedding_is_near_chance():
    X, _, rng = _data(n=150)
    R = rng.normal(size=(150, 2))
    assert abs(rnx_auc(X, R)) < 0.05
    assert abs(lcmc(X, R, 10)) < 0.05
    assert trustworthiness(X, R, 10) == pytest.approx(0.5, abs=0.08)


def test_better_embedding_ranks_higher():
    # Points on a noisy 2-D plane inside 10-D: PCA recovers it, a random
    # projection onto noise directions does not.
    rng = np.random.default_rng(3)
    latent = rng.uniform(-1, 1, size=(150, 2))
    X = np.hstack([latent * 5.0, 0.05 * rng.normal(size=(150, 8))])
    good, _ = pca_svd(pd.DataFrame(X), n_components=2)
    bad = X[:, 2:4]
    q_good = embedding_quality(X, good, n_neighbors=10)
    q_bad = embedding_quality(X, bad, n_neighbors=10)
    for key in ("trustworthiness", "continuity", "neighborhood_preservation", "lcmc", "rnx_auc"):
        assert q_good[key] > q_bad[key]
    assert q_good["trustworthiness"] > 0.95


def test_embedding_quality_agrees_with_individual_metrics():
    X, Y, _ = _data(n=80)
    q = embedding_quality(pd.DataFrame(X), pd.DataFrame(Y), n_neighbors=7)
    assert q["trustworthiness"] == pytest.approx(trustworthiness(X, Y, 7))
    assert q["continuity"] == pytest.approx(continuity(X, Y, 7))
    assert q["neighborhood_preservation"] == pytest.approx(neighborhood_preservation(X, Y, 7))
    assert q["lcmc"] == pytest.approx(lcmc(X, Y, 7))
    assert q["rnx_auc"] == pytest.approx(rnx_auc(X, Y))


def test_neighborhood_preservation_hand_example():
    # Tie-free 1-D input with growing gaps: the 1-NN of each point is unique.
    X = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])[:, None]
    # Input 1-NN sets: {1}, {0}, {1}, {2}, {3}, {4}.
    # The embedding moves point 5 next to point 0, between 0 and 1:
    Y = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 0.4])[:, None]
    # Embedding 1-NN sets: {5}, {5}, {1}, {2}, {3}, {0}.
    matches = [0, 0, 1, 1, 1, 0]
    assert neighborhood_preservation(X, Y, 1) == pytest.approx(np.mean(matches))
    assert lcmc(X, Y, 1) == pytest.approx(np.mean(matches) - 1.0 / 5.0)


def test_validation():
    X, Y, _ = _data(n=20)
    with pytest.raises(ValueError):
        trustworthiness(X, Y, 10)  # k must be < n / 2
    with pytest.raises(ValueError):
        trustworthiness(X, Y[:-1], 3)
    with pytest.raises(ValueError):
        trustworthiness(X, Y, 0)
    with pytest.raises(ValueError):
        neighborhood_preservation(X, Y, 20)
    with pytest.raises(ValueError):
        coranking_matrix(X[:3], Y[:3])
    with pytest.raises(ValueError):
        coranking_matrix(np.full((5, 2), np.nan), np.zeros((5, 2)))
    q = embedding_quality(X, Y, n_neighbors=12)
    assert np.isnan(q["trustworthiness"]) and np.isfinite(q["neighborhood_preservation"])


def test_pipeline_reports_quality(tmp_path):
    from pipeline import run_pipeline

    summary = run_pipeline(output_dir=tmp_path, n_samples=70, n_features=10)
    res = summary["results"]
    for name in ("pca_svd", "isomap", "ltsa", "tsne_classic"):
        assert 0.0 <= res[name]["trustworthiness"] <= 1.0
        assert 0.0 <= res[name]["continuity"] <= 1.0
        assert res[name]["rnx_auc"] <= 1.0
    assert "ltsa" in res and "variance" in res["ltsa"]
