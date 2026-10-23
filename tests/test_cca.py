"""Tests for two-view methods: CCA, PLS-SVD and PLS regression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn import cross_decomposition as skcd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cca import CCA, PLSSVD, PLSRegression, canonical_correlations  # noqa: E402


def _two_views(n=300, p=5, q=4, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    X = Z @ rng.normal(size=(2, p)) + rng.normal(size=(n, p))
    Y = Z @ rng.normal(size=(2, q)) + rng.normal(size=(n, q))
    return X, Y


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


class TestCCA:
    def test_correlations_match_variates(self):
        X, Y = _two_views()
        cca = CCA(n_components=3).fit(X, Y)
        Tx, Ty = cca.transform(X, Y)
        for j in range(3):
            assert _corr(Tx[:, j], Ty[:, j]) == pytest.approx(cca.correlations_[j], abs=1e-10)
        assert np.all(np.diff(cca.correlations_) <= 1e-12)
        np.testing.assert_allclose(cca.score(X, Y), cca.correlations_, atol=1e-10)

    def test_variates_whitened_within_view(self):
        X, Y = _two_views(seed=1)
        Tx, Ty = CCA(n_components=3).fit_transform(X, Y)
        np.testing.assert_allclose(np.cov(Tx, rowvar=False), np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.cov(Ty, rowvar=False), np.eye(3), atol=1e-10)
        # cross-covariance is diagonal: pair j only correlates with pair j
        cross = (Tx - Tx.mean(0)).T @ (Ty - Ty.mean(0)) / (len(Tx) - 1)
        np.testing.assert_allclose(cross - np.diag(np.diag(cross)), 0.0, atol=1e-10)

    def test_matches_sklearn_nipals(self):
        X, Y = _two_views(seed=2)
        ours = CCA(n_components=3).fit(X, Y).correlations_
        sk = skcd.CCA(n_components=3, scale=False, max_iter=5000, tol=1e-12).fit(X, Y)
        tx, ty = sk.transform(X, Y)
        theirs = [_corr(tx[:, j], ty[:, j]) for j in range(3)]
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_first_pair_beats_random_directions(self):
        X, Y = _two_views(seed=3)
        rho = CCA(n_components=1).fit(X, Y).correlations_[0]
        rng = np.random.default_rng(0)
        for _ in range(300):
            a, b = rng.normal(size=X.shape[1]), rng.normal(size=Y.shape[1])
            assert abs(_corr(X @ a, Y @ b)) <= rho + 1e-12

    def test_single_y_column_is_multiple_correlation(self):
        X, Y = _two_views(seed=4)
        y = Y[:, 0]
        Xd = np.c_[np.ones(len(X)), X]
        fitted = Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]
        r = _corr(fitted, y)
        assert canonical_correlations(X, y)[0] == pytest.approx(r, abs=1e-10)

    def test_affine_invariance_of_correlations(self):
        X, Y = _two_views(seed=5)
        rng = np.random.default_rng(1)
        Ax = rng.normal(size=(5, 5))
        Ay = rng.normal(size=(4, 4))
        base = canonical_correlations(X, Y)
        moved = canonical_correlations(X @ Ax + 3.0, Y @ Ay - 7.0)
        np.testing.assert_allclose(base, moved, atol=1e-9)
        scaled = CCA(n_components=4, scale=True).fit(X, Y).correlations_
        np.testing.assert_allclose(base, scaled, atol=1e-10)

    def test_independent_views_have_small_correlation(self):
        rng = np.random.default_rng(6)
        rho = canonical_correlations(rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3)))
        assert rho[0] < 0.1

    def test_regularisation_shrinks_and_allows_wide_data(self):
        X, Y = _two_views(seed=7)
        plain = CCA(n_components=2).fit(X, Y).correlations_
        ridge = CCA(n_components=2, reg=1.0).fit(X, Y).correlations_
        assert np.all(ridge <= plain + 1e-12)
        rng = np.random.default_rng(2)
        Xw, Yw = rng.normal(size=(20, 30)), rng.normal(size=(20, 25))
        with pytest.raises(ValueError, match="singular"):
            CCA(n_components=2).fit(Xw, Yw)
        out = CCA(n_components=2, reg=0.5).fit(Xw, Yw).correlations_
        assert np.all((out >= 0) & (out <= 1))

    def test_out_of_sample_and_loadings(self):
        X, Y = _two_views(n=600, seed=8)
        cca = CCA(n_components=2).fit(X[:400], Y[:400])
        held = cca.score(X[400:], Y[400:])
        assert held[0] > 0.5
        assert cca.transform(X[400:]).shape == (200, 2)
        assert cca.x_loadings_.shape == (5, 2) and cca.y_loadings_.shape == (4, 2)
        assert np.all(np.abs(cca.x_loadings_) <= 1 + 1e-12)

    def test_dataframe_input(self):
        X, Y = _two_views(seed=9)
        a = CCA(2).fit(pd.DataFrame(X), pd.DataFrame(Y)).correlations_
        b = CCA(2).fit(X, Y).correlations_
        np.testing.assert_allclose(a, b)

    def test_validation(self):
        X, Y = _two_views()
        with pytest.raises(ValueError):
            CCA(n_components=5).fit(X, Y)
        with pytest.raises(ValueError):
            CCA(n_components=0).fit(X, Y)
        with pytest.raises(ValueError):
            CCA(reg=-1.0).fit(X, Y)
        with pytest.raises(ValueError, match="same number"):
            CCA().fit(X, Y[:-1])
        with pytest.raises(RuntimeError):
            CCA().transform(X)
        with pytest.raises(ValueError, match="different number"):
            CCA(2).fit(X, Y).transform(X[:, :3])


class TestPLSSVD:
    @pytest.mark.parametrize("scale", [True, False])
    def test_matches_sklearn(self, scale):
        X, Y = _two_views(seed=10)
        ours = PLSSVD(3, scale=scale).fit(X, Y)
        sk = skcd.PLSSVD(3, scale=scale).fit(X, Y)
        np.testing.assert_allclose(np.abs(ours.x_weights_), np.abs(sk.x_weights_), atol=1e-10)
        tx, ty = ours.transform(X, Y)
        sx, sy = sk.transform(X, Y)
        np.testing.assert_allclose(np.abs(tx), np.abs(sx), atol=1e-9)
        np.testing.assert_allclose(np.abs(ty), np.abs(sy), atol=1e-9)

    def test_covariances_are_score_covariances(self):
        X, Y = _two_views(seed=11)
        pls = PLSSVD(3, scale=False).fit(X, Y)
        tx, ty = pls.transform(X, Y)
        for j in range(3):
            assert np.cov(tx[:, j], ty[:, j])[0, 1] == pytest.approx(pls.covariances_[j], rel=1e-10)
        np.testing.assert_allclose(pls.x_weights_.T @ pls.x_weights_, np.eye(3), atol=1e-12)


class TestPLSRegression:
    @pytest.mark.parametrize("scale", [True, False])
    def test_pls2_matches_sklearn(self, scale):
        X, Y = _two_views(seed=12)
        ours = PLSRegression(3, scale=scale).fit(X, Y)
        sk = skcd.PLSRegression(3, scale=scale, max_iter=5000, tol=1e-14).fit(X, Y)
        np.testing.assert_allclose(ours.predict(X), sk.predict(X), atol=1e-6)
        np.testing.assert_allclose(np.abs(ours.transform(X)), np.abs(sk.transform(X)), atol=1e-5)
        assert ours.score(X, Y) == pytest.approx(sk.score(X, Y), abs=1e-8)

    def test_pls1_matches_sklearn_exactly(self):
        X, Y = _two_views(seed=13)
        y = Y[:, 1]
        ours = PLSRegression(2).fit(X, y)
        sk = skcd.PLSRegression(2).fit(X, y)
        pred = ours.predict(X)
        assert pred.shape == (len(y),)
        np.testing.assert_allclose(pred, np.ravel(sk.predict(X)), atol=1e-10)

    def test_full_rank_equals_ols(self):
        X, Y = _two_views(seed=14)
        pls = PLSRegression(n_components=X.shape[1], scale=False).fit(X, Y)
        Xd = np.c_[np.ones(len(X)), X]
        beta = np.linalg.lstsq(Xd, Y, rcond=None)[0]
        np.testing.assert_allclose(pls.predict(X), Xd @ beta, atol=1e-8)
        np.testing.assert_allclose(pls.coef_, beta[1:], atol=1e-8)
        np.testing.assert_allclose(pls.intercept_, beta[0], atol=1e-8)

    def test_scores_orthogonal_and_rotations_reproduce_them(self):
        X, Y = _two_views(seed=15)
        pls = PLSRegression(3).fit(X, Y)
        T = pls.x_scores_
        G = T.T @ T
        np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-8)
        np.testing.assert_allclose(pls.transform(X), T, atol=1e-10)

    def test_r2_increases_with_components(self):
        X, Y = _two_views(seed=16)
        r2 = [PLSRegression(k).fit(X, Y).score(X, Y) for k in range(1, 6)]
        assert all(b >= a - 1e-12 for a, b in zip(r2, r2[1:]))

    def test_validation(self):
        X, Y = _two_views()
        with pytest.raises(ValueError):
            PLSRegression(n_components=6).fit(X, Y)
        with pytest.raises(RuntimeError):
            PLSRegression().predict(X)
        with pytest.raises(ValueError, match="finite"):
            PLSRegression().fit(np.where(X > 2, np.nan, X), Y)
