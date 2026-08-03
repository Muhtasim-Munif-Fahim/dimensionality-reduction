#!/usr/bin/env python3
"""Run dimensionality reduction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import run_pipeline


def main() -> int:
    print("=" * 60)
    print("Dimensionality Reduction Pipeline")
    print("=" * 60)

    results = run_pipeline(output_dir="output", n_samples=400, n_features=40)

    print(f"\nData: {results['n_samples']} samples x {results['n_features']} features\n")
    print(f"{'Method':<18} {'Silhouette':<12} {'Separation':<12} {'Variance':<10}")
    print("-" * 52)
    for name, metrics in results["results"].items():
        sil = f"{metrics['silhouette']:.4f}" if isinstance(metrics.get("silhouette"), float) else "n/a"
        sep = f"{metrics['separation']:.4f}" if isinstance(metrics.get("separation"), float) else "n/a"
        var = f"{metrics['variance']:.4f}" if isinstance(metrics.get("variance"), float) else "n/a"
        print(f"{name:<18} {sil:<12} {sep:<12} {var:<10}")

    print(f"\nResults saved to output/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
