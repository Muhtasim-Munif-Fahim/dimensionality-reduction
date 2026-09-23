"""Tests for FastICA on linear mixtures of independent sources."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ica import FastICA


def _mixed_signals(
    n_samples: int = 1200,
    n_features: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three independent sources observed through a random mixing matrix.

    Returns ``(observed, sources, mixing)`` with ``observed = sources @ mixing.T``.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, n_samples, endpoint=False)
    sine = np.sin(2 * np.pi * 0.7 * t)
    square = np.sign(np.sin(2 * np.pi * 1.5 * t + 0.3))
    laplace = rng.laplace(size=n_samples)
    sources = np.column_stack([sine, square, laplace])
    sources -= sources.mean(axis=0)
    sources /= sources.std(axis=0)
    mixing = rng.normal(size=(n_features, sources.shape[1]))
    observed = sources @ mixing.T
    return observed, sources, mixing


def _abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return abs(float(np.dot(left, right) / denom))


def _matched_correlations(estimated: np.ndarray, true: np.ndarray) -> list[float]:
    """Greedy absolute correlations after an unknown permutation."""
    used: set[int] = set()
    scores: list[float] = []
    for column in range(estimated.shape[1]):
        candidates = [
            (_abs_corr(estimated[:, column], true[:, source]), source)
            for source in range(true.shape[1])
            if source not in used
        ]
        score, source = max(candidates)
        used.add(source)
        scores.append(score)
    return scores


def _pca_scores(observed: np.ndarray, n_components: int) -> np.ndarray:
    centered = observed - observed.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


class TestMixedSignals:
    def test_recovers_sources_up_to_sign_and_order(self):
        for seed in range(5):
            observed, sources, _ = _mixed_signals(seed=seed)
            recovered = FastICA(n_components=3, random_state=seed).fit_transform(observed)
            assert min(_matched_correlations(recovered, sources)) > 0.98

    def test_matches_sources_better_than_pca(self):
        for seed in range(5):
            observed, sources, _ = _mixed_signals(seed=seed)
            recovered = FastICA(n_components=3, random_state=0).fit_transform(observed)
            ica_score = min(_matched_correlations(recovered, sources))
            pca_score = min(_matched_correlations(_pca_scores(observed, 3), sources))
            assert ica_score > pca_score
            assert ica_score > 0.98

    def test_inverse_reconstructs_noiseless_mixture(self):
        observed, _, _ = _mixed_signals()
        ica = FastICA(n_components=3, random_state=0).fit(observed)
        restored = ica.inverse_transform(ica.transform(observed))
        assert restored.shape == observed.shape
        assert np.allclose(restored, observed, atol=1e-8)

    def test_held_out_samples_use_the_fitted_unmixing(self):
        observed, _, mixing = _mixed_signals(seed=1)
        ica = FastICA(n_components=3, random_state=1).fit(observed)
        rng = np.random.default_rng(99)
        t = np.linspace(0, 4, 400, endpoint=False)
        sources = np.column_stack(
            [
                np.sin(2 * np.pi * 0.7 * t),
                np.sign(np.sin(2 * np.pi * 1.5 * t + 0.3)),
                rng.laplace(size=400),
            ]
        )
        sources -= sources.mean(axis=0)
        sources /= sources.std(axis=0)
        held_out = sources @ mixing.T
        recovered = ica.transform(held_out)
        assert recovered.shape == (400, 3)
        assert min(_matched_correlations(recovered, sources)) > 0.95

    def test_components_and_mixing_shapes(self):
        observed, _, _ = _mixed_signals(n_features=5)
        ica = FastICA(n_components=3, random_state=0).fit(observed)
        assert ica.components_.shape == (3, 5)
        assert ica.mixing_.shape == (5, 3)
        assert ica.mean_.shape == (5,)
        assert ica.n_components_ == 3
        assert ica.n_features_in_ == 5
        assert ica.n_iter_ < 200
        gram = ica.components_ @ ica.mixing_
        assert np.allclose(gram, np.eye(3), atol=1e-8)

    def test_reproducible_given_random_state(self):
        observed, _, _ = _mixed_signals()
        first = FastICA(n_components=3, random_state=7).fit_transform(observed)
        second = FastICA(n_components=3, random_state=7).fit_transform(observed)
        assert np.allclose(first, second)

    def test_fit_does_not_mutate_input(self):
        observed, _, _ = _mixed_signals(n_samples=300)
        before = observed.copy()
        FastICA(n_components=3, random_state=0).fit(observed)
        assert np.array_equal(observed, before)

    def test_dataframe_matches_ndarray(self):
        observed, _, _ = _mixed_signals(n_samples=500, n_features=4)
        frame = pd.DataFrame(observed, columns=[f"f{i}" for i in range(observed.shape[1])])
        from_frame = FastICA(n_components=3, random_state=2).fit_transform(frame)
        from_array = FastICA(n_components=3, random_state=2).fit_transform(observed)
        assert from_frame.shape == (len(frame), 3)
        assert np.allclose(from_frame, from_array)

    def test_none_uses_every_feature_when_square(self):
        observed, sources, _ = _mixed_signals(n_features=3, seed=4)
        ica = FastICA(n_components=None, random_state=4).fit(observed)
        assert ica.n_components_ == 3
        recovered = ica.transform(observed)
        assert min(_matched_correlations(recovered, sources)) > 0.98

    def test_fit_returns_self(self):
        observed, _, _ = _mixed_signals(n_samples=200)
        ica = FastICA(n_components=2, random_state=0)
        assert ica.fit(observed) is ica


class TestFastICAValidation:
    def test_rejects_bad_arguments(self):
        _raises(ValueError, lambda: FastICA(n_components=0))
        _raises(ValueError, lambda: FastICA(n_components=-2))
        _raises(ValueError, lambda: FastICA(n_components=1.5))
        _raises(ValueError, lambda: FastICA(n_components=True))
        _raises(ValueError, lambda: FastICA(max_iter=0))
        _raises(ValueError, lambda: FastICA(tol=0))
        _raises(ValueError, lambda: FastICA(tol=-1e-3))
        _raises(ValueError, lambda: FastICA(random_state=1.2))

    def test_rejects_bad_shapes_and_rank(self):
        observed, _, _ = _mixed_signals(n_samples=40, n_features=5)
        _raises(ValueError, lambda: FastICA(n_components=6).fit(observed))
        _raises(ValueError, lambda: FastICA(n_components=None).fit(observed[:4]))
        _raises(ValueError, lambda: FastICA(n_components=1).fit(np.ones((10, 3))))
        _raises(ValueError, lambda: FastICA(n_components=4).fit(observed))
        broken = observed.copy()
        broken[0, 0] = np.nan
        _raises(ValueError, lambda: FastICA(n_components=2).fit(broken))
        _raises(ValueError, lambda: FastICA(n_components=2).fit(observed[:, 0]))

    def test_rejects_unfitted_and_wrong_width(self):
        observed, _, _ = _mixed_signals(n_samples=80, n_features=5)
        ica = FastICA(n_components=2, random_state=0).fit(observed)
        _raises(RuntimeError, lambda: FastICA(n_components=2).transform(observed))
        _raises(ValueError, lambda: ica.transform(observed[:, :3]))
        _raises(ValueError, lambda: ica.inverse_transform(np.ones((4, 1))))
        _raises(ValueError, lambda: ica.inverse_transform(np.ones(4)))
