"""Tests for Procrustes alignment, disparity, PROTEST and GPA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from procrustes import (
    Procrustes,
    generalized_procrustes,
    orthogonal_procrustes,
    procrustes_disparity,
    protest,
)


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_recovers_similarity_transform_exactly():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 2))
    R = _rotation(0.7)
    B = 2.5 * A @ R + np.array([3.0, -1.0])
    model = Procrustes().fit(A, B)
    np.testing.assert_allclose(model.rotation_, R, atol=1e-10)
    assert model.scale_ == pytest.approx(2.5)
    assert model.rmsd_ == pytest.approx(0.0, abs=1e-10)
    assert model.disparity_ == pytest.approx(0.0, abs=1e-12)
    new = rng.normal(size=(5, 2))
    np.testing.assert_allclose(model.transform(new), 2.5 * new @ R + [3.0, -1.0], atol=1e-10)


def test_reflection_handling():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 2))
    B = A * np.array([1.0, -1.0])  # mirror image
    assert Procrustes().fit(A, B).rmsd_ == pytest.approx(0.0, abs=1e-10)
    rigid = Procrustes(allow_reflection=False).fit(A, B)
    assert np.linalg.det(rigid.rotation_) == pytest.approx(1.0)
    assert rigid.rmsd_ > 0.1


def test_disparity_matches_scipy_and_is_symmetric():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(50, 3))
    B = A @ np.linalg.qr(rng.normal(size=(3, 3)))[0] + 0.3 * rng.normal(size=(50, 3))
    _, _, ref = scipy_procrustes(A, B)
    assert procrustes_disparity(A, B) == pytest.approx(ref, rel=1e-9)
    assert procrustes_disparity(B, A) == pytest.approx(ref, rel=1e-9)
    assert 0.0 < ref < 1.0


def test_different_dimensions_are_zero_padded():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(20, 2))
    B = np.hstack([A @ _rotation(1.1), np.zeros((20, 1))])
    assert procrustes_disparity(A, B) == pytest.approx(0.0, abs=1e-12)
    model = Procrustes(scaling=False).fit(A, B)
    assert model.transform(A).shape == (20, 3)


def test_orthogonal_procrustes_is_orthogonal():
    rng = np.random.default_rng(4)
    R, _ = orthogonal_procrustes(rng.normal(size=(10, 4)), rng.normal(size=(10, 4)))
    np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-10)


def test_protest_detects_agreement_and_null():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(40, 2))
    related = A @ _rotation(0.4) + 0.1 * rng.normal(size=(40, 2))
    res = protest(A, related, n_permutations=199, random_state=0)
    assert res.p_value == pytest.approx(1 / 200)
    assert res.correlation > 0.95
    unrelated = rng.normal(size=(40, 2))
    assert protest(A, unrelated, n_permutations=199, random_state=0).p_value > 0.01


def test_generalized_procrustes_aligns_rotated_copies():
    rng = np.random.default_rng(6)
    base = rng.normal(size=(25, 2))
    configs = [3.0 * base @ _rotation(t) + rng.normal(size=2) for t in (0.0, 0.9, 2.2, -1.3)]
    res = generalized_procrustes(configs)
    assert res.converged
    assert res.residual == pytest.approx(0.0, abs=1e-10)
    for c in res.aligned:
        np.testing.assert_allclose(c, res.consensus, atol=1e-8)


def test_input_validation():
    with pytest.raises(ValueError):
        Procrustes().fit(np.ones((5, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError):
        Procrustes().fit(np.ones((5, 2)), np.random.default_rng(0).normal(size=(5, 2)))
    with pytest.raises(ValueError):
        generalized_procrustes([np.eye(3)])
