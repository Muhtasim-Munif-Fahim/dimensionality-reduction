"""Reduction method comparison summaries.

``format_results_table`` and ``best_method`` read whatever the pipeline
recorded. That includes ``nmf`` alongside PCA, LDA, t-SNE, and the autoencoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def results_to_dataframe(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["results"]).T
    return df.round(4)


def format_results_table(results: dict) -> str:
    df = results_to_dataframe(results)
    lines = [f"{'Method':<18} {'Silhouette':<12} {'Separation':<12} {'Variance':<10}"]
    lines.append("-" * 52)
    for name, row in df.iterrows():
        sil = f"{row.get('silhouette', float('nan')):.4f}"
        sep = f"{row.get('separation', float('nan')):.4f}"
        var = f"{row.get('variance', float('nan')):.4f}"
        lines.append(f"{name:<18} {sil:<12} {sep:<12} {var:<10}")
    return "\n".join(lines)


def best_method(results: dict, metric: str = "silhouette") -> str:
    best = None
    best_score = float("-inf")
    for name, metrics in results["results"].items():
        score = metrics.get(metric)
        if isinstance(score, float) and score > best_score:
            best_score = score
            best = name
    return best or "none"


def load_results(path: str | Path = "output/results.json") -> dict:
    return json.loads(Path(path).read_text())
