"""Tests for comparison module."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comparison import format_results_table, best_method


class TestComparison:
    def test_format_table(self):
        results = {"results": {"pca": {"silhouette": 0.8, "separation": 2.0, "variance": 0.5}}}
        table = format_results_table(results)
        assert "pca" in table
        assert "0.8" in table

    def test_best_method(self):
        results = {"results": {"a": {"silhouette": 0.5}, "b": {"silhouette": 0.9}}}
        assert best_method(results) == "b"
