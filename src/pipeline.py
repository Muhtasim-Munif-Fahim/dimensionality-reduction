"""Main pipeline: compare dimensionality reduction methods."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_generation import generate_high_dim_data
from pca import pca_manual, pca_svd, variance_retained
from nmf import nmf_reduce
from supervised_methods import lda_reduce, tsne_reduce
from tsne import TSNE
from isomap import Isomap
from lle import LLE
from mds import ClassicalMDS
from kernel_pca import KernelPCA
from spectral_embedding import SpectralEmbedding
from factor_analysis import FactorAnalysis
from diffusion_maps import DiffusionMaps
from autoencoder import NumpyAutoencoder
from evaluation import silhouette_of_embedding, cluster_separation, reconstruction_error


def run_pipeline(
    output_dir: str | Path = "output",
    n_samples: int = 400,
    n_features: int = 40,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df, labels = generate_high_dim_data(n_samples=n_samples, n_features=n_features)

    results = {}

    pca_manual_emb, pca_info = pca_manual(df, n_components=2)
    results["pca_manual"] = {
        "silhouette": round(silhouette_of_embedding(pca_manual_emb, labels), 4),
        "separation": round(cluster_separation(pca_manual_emb, labels), 4),
        "variance": round(pca_info["cumulative_variance"], 4),
    }

    pca_svd_emb, svd_info = pca_svd(df, n_components=2)
    results["pca_svd"] = {
        "silhouette": round(silhouette_of_embedding(pca_svd_emb, labels), 4),
        "separation": round(cluster_separation(pca_svd_emb, labels), 4),
        "variance": round(svd_info["cumulative_variance"], 4),
    }

    lda_emb = lda_reduce(df, labels, n_components=2)
    results["lda"] = {
        "silhouette": round(silhouette_of_embedding(lda_emb, labels), 4),
        "separation": round(cluster_separation(lda_emb, labels), 4),
        "variance": float("nan"),
    }

    tsne_emb = tsne_reduce(df, n_components=2)
    results["tsne"] = {
        "silhouette": round(silhouette_of_embedding(tsne_emb, labels), 4),
        "separation": round(cluster_separation(tsne_emb, labels), 4),
        "variance": float("nan"),
    }

    # Classic NumPy t-SNE (exact algorithm). Keep perplexity small for the
    # demo sample size and cap iterations so the pipeline stays interactive.
    classic = TSNE(
        n_components=2,
        perplexity=min(30.0, max(5.0, n_samples / 4.0)),
        n_iter=300,
        random_state=0,
    ).fit_transform(df.to_numpy())
    classic_emb = pd.DataFrame(classic, columns=["TSNE1", "TSNE2"])
    results["tsne_classic"] = {
        "silhouette": round(silhouette_of_embedding(classic_emb, labels), 4),
        "separation": round(cluster_separation(classic_emb, labels), 4),
        "variance": float("nan"),
    }


    isomap_emb = Isomap(
        n_components=2,
        n_neighbors=min(10, max(3, n_samples // 20)),
    ).fit_transform(df.to_numpy())
    isomap_frame = pd.DataFrame(isomap_emb, columns=["ISO1", "ISO2"])
    results["isomap"] = {
        "silhouette": round(silhouette_of_embedding(isomap_frame, labels), 4),
        "separation": round(cluster_separation(isomap_frame, labels), 4),
        "variance": float("nan"),
    }

    lle_emb = LLE(
        n_components=2,
        n_neighbors=min(10, max(3, n_samples // 20)),
    ).fit_transform(df.to_numpy())
    lle_frame = pd.DataFrame(lle_emb, columns=["LLE1", "LLE2"])
    results["lle"] = {
        "silhouette": round(silhouette_of_embedding(lle_frame, labels), 4),
        "separation": round(cluster_separation(lle_frame, labels), 4),
        "variance": float("nan"),
    }

    se_emb = SpectralEmbedding(
        n_components=2,
        n_neighbors=min(10, max(3, n_samples // 20)),
    ).fit_transform(df.to_numpy())
    se_frame = pd.DataFrame(se_emb, columns=["SE1", "SE2"])
    results["spectral_embedding"] = {
        "silhouette": round(silhouette_of_embedding(se_frame, labels), 4),
        "separation": round(cluster_separation(se_frame, labels), 4),
        "variance": float("nan"),
    }

    fa_emb = FactorAnalysis(
        n_components=2,
        max_iter=200,
        tol=1e-3,
        random_state=0,
    ).fit_transform(df.to_numpy())
    fa_frame = pd.DataFrame(fa_emb, columns=["FA1", "FA2"])
    results["factor_analysis"] = {
        "silhouette": round(silhouette_of_embedding(fa_frame, labels), 4),
        "separation": round(cluster_separation(fa_frame, labels), 4),
        "variance": float("nan"),
    }

    dm_emb = DiffusionMaps(
        n_components=2,
        n_neighbors=min(10, max(3, n_samples // 20)),
        t=1.0,
        alpha=0.0,
    ).fit_transform(df.to_numpy())
    dm_frame = pd.DataFrame(dm_emb, columns=["DM1", "DM2"])
    results["diffusion_maps"] = {
        "silhouette": round(silhouette_of_embedding(dm_frame, labels), 4),
        "separation": round(cluster_separation(dm_frame, labels), 4),
        "variance": float("nan"),
    }

    mds_emb = ClassicalMDS(n_components=2).fit_transform(df.to_numpy())
    mds_frame = pd.DataFrame(mds_emb, columns=["MDS1", "MDS2"])
    results["classical_mds"] = {
        "silhouette": round(silhouette_of_embedding(mds_frame, labels), 4),
        "separation": round(cluster_separation(mds_frame, labels), 4),
        "variance": float("nan"),
    }

    ae = NumpyAutoencoder(n_input=n_features, n_hidden=8, learning_rate=0.005)
    ae.fit(df.to_numpy(), epochs=50)
    results["autoencoder"] = {
        "reconstruction_error": round(ae.reconstruct_error(df.to_numpy()), 6),
        "silhouette": round(silhouette_of_embedding(pd.DataFrame(ae.encode(df.to_numpy()), columns=[f"h{i}" for i in range(8)]), labels), 4),
        "variance": float("nan"),
    }

    # Synthetic features are signed Gaussians. NMF needs non-negative columns,
    # so shift each column until its minimum is zero.
    nonneg = df - df.min(axis=0)
    nmf_emb, nmf_info = nmf_reduce(nonneg, n_components=2, random_state=0)
    results["nmf"] = {
        "silhouette": round(silhouette_of_embedding(nmf_emb, labels), 4),
        "separation": round(cluster_separation(nmf_emb, labels), 4),
        "variance": float("nan"),
        "reconstruction_error": round(float(nmf_info["reconstruction_error"]), 6),
    }

    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_csv(output / "high_dim_data.csv", index=False)
    return summary
