"""Tests for non-negative matrix factorization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comparison import format_results_table
from nmf import NMF, nmf_reduce
from pipeline import run_pipeline


def _topic_matrix(
    n_docs: int = 80,
    n_terms: int = 16,
    n_topics: int = 4,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Non-negative documents generated from a handful of topics."""
    rng = np.random.default_rng(seed)
    topics = rng.random((n_topics, n_terms))
    topics[topics < 0.55] = 0.0
    topics += 0.05
    weights = rng.dirichlet(np.ones(n_topics) * 0.35, size=n_docs)
    documents = weights @ topics
    if noise:
        documents = documents + noise * rng.random(documents.shape)
    return documents


def _random_factor_error(matrix: np.ndarray, n_components: int, rng: np.random.Generator) -> float:
    n_samples, n_features = matrix.shape
    scale = np.sqrt(max(float(matrix.mean()), 1e-12) / n_components)
    codes = rng.random((n_samples, n_components)) * scale
    components = rng.random((n_components, n_features)) * scale
    residual = matrix - codes @ components
    return float(np.mean(residual * residual))


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


class TestNMFReduce:
    def test_shape_and_nonnegative_factors(self):
        documents = _topic_matrix()
        embedding, info = nmf_reduce(documents, n_components=4, random_state=0)
        components = info["components"]
        assert embedding.shape == (documents.shape[0], 4)
        assert components.shape == (4, documents.shape[1])
        assert isinstance(embedding, np.ndarray)
        assert np.all(embedding >= 0.0)
        assert np.all(components >= 0.0)
        assert info["n_iter"] >= 1
        assert info["reconstruction_error"] >= 0.0

    def test_reconstruction_beats_random_factors(self):
        documents = _topic_matrix(seed=1)
        embedding, info = nmf_reduce(
            documents,
            n_components=4,
            max_iter=300,
            tol=1e-8,
            random_state=0,
        )
        restored = np.asarray(embedding) @ info["components"]
        fitted = float(np.mean((documents - restored) ** 2))
        assert abs(fitted - info["reconstruction_error"]) < 1e-8
        rng = np.random.default_rng(1)
        random_errors = [_random_factor_error(documents, 4, rng) for _ in range(8)]
        assert fitted < min(random_errors)

    def test_updates_improve_a_random_start(self):
        documents = _topic_matrix(seed=2)
        short = NMF(
            n_components=4,
            init="random",
            max_iter=1,
            tol=1e-15,
            random_state=3,
        ).fit(documents)
        long = NMF(
            n_components=4,
            init="random",
            max_iter=80,
            tol=1e-12,
            random_state=3,
        ).fit(documents)
        assert long.reconstruction_err_ < short.reconstruction_err_
        assert np.all(long.components_ >= 0.0)
        assert np.all(long.fit_transform(documents) >= 0.0)

    def test_low_rank_reconstruction_is_small(self):
        rng = np.random.default_rng(4)
        codes = rng.random((50, 3)) + 0.2
        basis = rng.random((3, 9)) + 0.2
        documents = codes @ basis
        _, info = nmf_reduce(
            documents,
            n_components=3,
            max_iter=400,
            tol=1e-10,
            random_state=0,
        )
        relative = info["reconstruction_error"] / float(np.mean(documents**2))
        assert relative < 1e-3

    def test_more_components_lower_reconstruction_error(self):
        documents = _topic_matrix(noise=0.02, seed=5)
        errors = []
        for rank in (1, 2, 4):
            _, info = nmf_reduce(
                documents,
                n_components=rank,
                max_iter=250,
                tol=1e-8,
                random_state=0,
            )
            errors.append(info["reconstruction_error"])
        assert errors[0] > errors[1] > errors[2]

    def test_dataframe_preserves_index_and_names_columns(self):
        documents = _topic_matrix(n_docs=30, n_terms=8, n_topics=3, seed=6)
        frame = pd.DataFrame(
            documents,
            index=np.arange(100, 130),
            columns=[f"t{i}" for i in range(documents.shape[1])],
        )
        embedding, info = nmf_reduce(frame, n_components=3, random_state=1)
        assert isinstance(embedding, pd.DataFrame)
        assert list(embedding.columns) == ["NMF1", "NMF2", "NMF3"]
        assert embedding.index.equals(frame.index)
        assert embedding.shape == (30, 3)
        assert np.all(embedding.to_numpy() >= 0.0)
        assert info["components"].shape == (3, 8)

    def test_dataframe_matches_ndarray(self):
        documents = _topic_matrix(n_docs=40, n_terms=7, n_topics=2, seed=7)
        frame = pd.DataFrame(documents)
        from_frame, frame_info = nmf_reduce(frame, n_components=2, random_state=0)
        from_array, array_info = nmf_reduce(documents, n_components=2, random_state=0)
        assert np.allclose(from_frame.to_numpy(), from_array)
        assert np.allclose(frame_info["components"], array_info["components"])

    def test_reproducible_given_random_state(self):
        documents = _topic_matrix(seed=8)
        first, first_info = nmf_reduce(
            documents, n_components=3, init="random", random_state=11
        )
        second, second_info = nmf_reduce(
            documents, n_components=3, init="random", random_state=11
        )
        assert np.allclose(first, second)
        assert np.allclose(first_info["components"], second_info["components"])

    def test_fit_does_not_mutate_input(self):
        documents = _topic_matrix(n_docs=25, n_terms=6, n_topics=2)
        before = documents.copy()
        NMF(n_components=2, random_state=0).fit(documents)
        assert np.array_equal(documents, before)

    def test_zero_matrix_stays_nonnegative_and_exact(self):
        zeros = np.zeros((12, 5))
        embedding, info = nmf_reduce(zeros, n_components=2, random_state=0)
        assert embedding.shape == (12, 2)
        assert np.all(embedding == 0.0)
        assert np.all(info["components"] == 0.0)
        assert info["reconstruction_error"] == 0.0

    def test_inverse_transform_shape(self):
        documents = _topic_matrix(n_docs=20, n_terms=9, n_topics=3, seed=9)
        model = NMF(n_components=3, random_state=0)
        codes = model.fit_transform(documents)
        restored = model.inverse_transform(codes)
        assert codes.shape == (20, 3)
        assert restored.shape == documents.shape
        assert np.all(codes >= 0.0)
        assert np.all(model.components_ >= 0.0)
        assert np.all(restored >= -1e-8)
        assert model.n_components_ == 3
        assert model.n_features_in_ == 9

    def test_transform_held_out_rows_are_nonnegative(self):
        documents = _topic_matrix(seed=10)
        model = NMF(n_components=4, random_state=0).fit(documents[:60])
        codes = model.transform(documents[60:])
        assert codes.shape == (documents.shape[0] - 60, 4)
        assert np.all(codes >= 0.0)
        restored = model.inverse_transform(codes)
        assert restored.shape == documents[60:].shape

    def test_sparse_matches_dense(self):
        documents = _topic_matrix(n_docs=35, n_terms=10, n_topics=3, seed=12)
        dense = NMF(n_components=3, random_state=0).fit_transform(documents)
        sparse_codes = NMF(n_components=3, random_state=0).fit_transform(
            sparse.csr_matrix(documents)
        )
        assert np.allclose(dense, sparse_codes, atol=1e-8)

    def test_fit_returns_self(self):
        documents = _topic_matrix(n_docs=15, n_terms=5, n_topics=2)
        model = NMF(n_components=2, random_state=0)
        assert model.fit(documents) is model

    def test_fit_does_not_mutate_sparse_values(self):
        documents = sparse.csr_matrix(_topic_matrix(n_docs=18, n_terms=6, n_topics=2))
        before = documents.data.copy()
        NMF(n_components=2, random_state=0).fit(documents)
        assert np.array_equal(documents.data, before)


class TestPipeline:
    def test_pipeline_records_nmf(self, tmp_path):
        summary = run_pipeline(output_dir=tmp_path, n_samples=70, n_features=10)
        metrics = summary["results"]["nmf"]
        assert metrics["reconstruction_error"] >= 0.0
        assert -1.0 <= metrics["silhouette"] <= 1.0
        assert metrics["separation"] >= 0.0
        assert "nmf" in format_results_table(summary)


class TestNMFValidation:
    def test_rejects_bad_arguments(self):
        _raises(ValueError, lambda: NMF(n_components=0))
        _raises(ValueError, lambda: NMF(n_components=-2))
        _raises(ValueError, lambda: NMF(n_components=1.5))
        _raises(ValueError, lambda: NMF(n_components=True))
        _raises(ValueError, lambda: NMF(max_iter=0))
        _raises(ValueError, lambda: NMF(tol=0))
        _raises(ValueError, lambda: NMF(tol=-1e-3))
        _raises(ValueError, lambda: NMF(random_state=1.2))
        _raises(ValueError, lambda: NMF(init="svd"))

    def test_rejects_negative_nonfinite_and_rank(self):
        documents = _topic_matrix(n_docs=20, n_terms=5, n_topics=2)
        _raises(ValueError, lambda: NMF(n_components=6).fit(documents))
        negative = documents.copy()
        negative[0, 0] = -0.1
        _raises(ValueError, lambda: NMF(n_components=2).fit(negative))
        broken = documents.copy()
        broken[0, 1] = np.nan
        _raises(ValueError, lambda: NMF(n_components=2).fit(broken))
        _raises(ValueError, lambda: NMF(n_components=2).fit(documents[:, 0]))
        _raises(ValueError, lambda: nmf_reduce([[-1.0, 2.0], [0.0, 1.0]], n_components=1))

    def test_rejects_unfitted_and_wrong_width(self):
        documents = _topic_matrix(n_docs=16, n_terms=5, n_topics=2)
        model = NMF(n_components=2, random_state=0).fit(documents)
        _raises(RuntimeError, lambda: NMF(n_components=2).transform(documents))
        _raises(ValueError, lambda: model.transform(documents[:, :3]))
        _raises(ValueError, lambda: model.inverse_transform(np.ones((4, 1))))
        _raises(ValueError, lambda: model.inverse_transform(np.ones(4)))
