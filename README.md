# Dimensionality Reduction

Practice project comparing dimensionality reduction methods for high-dimensional data.

## Project Structure

```
src/                - Source code
data/               - Sample data
tests/              - Unit tests
```

## Methods

- PCA (manual implementation + scikit-learn)
- SVD-based reduction
- Truncated SVD for sparse TF-IDF matrices (no centering)
- FastICA for linear mixtures of independent sources
- NMF (non-negative parts-based factors, multiplicative updates)
- LDA (supervised)
- t-SNE (visualization; sklearn wrapper)
- Classic t-SNE (exact NumPy, perplexity / learning-rate)
- Isomap (geodesic kNN graph + classical MDS)
- LLE (Locally Linear Embedding; neighbourhood reconstruction)
- Spectral Embedding / Laplacian Eigenmaps (kNN Laplacian eigenvectors)
- Factor Analysis (EM / MLE latent factors with diagonal uniquenesses)
- Classical MDS (metric multidimensional scaling from pairwise distances)
- Sammon mapping (nonlinear metric MDS; fractional distance stress)
- Sparse PCA (L1 soft-thresholded loadings; Zou/Hastie/Tibshirani-style)
- Kernel PCA (RBF / linear / polynomial Gram + eigendecomposition)
- Autoencoder (numpy from scratch)
- LTSA (Local Tangent Space Alignment; neighbourhood PCA + global align)
- Embedding quality metrics: trustworthiness, continuity, co-ranking `Q_NX` / LCMC / `R_NX` AUC


## LTSA (Local Tangent Space Alignment)

LLE preserves reconstruction weights; Isomap preserves geodesic distances.
`LTSA` (Zhang & Zha, 2004) instead estimates a local tangent basis at each
point via neighbourhood PCA and aligns those tangent coordinates into a
global embedding. Classic LTSA is transductive (no out-of-sample
`transform`).

```python
import numpy as np
from ltsa import LTSA

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 8))
emb = LTSA(n_components=2, n_neighbors=10).fit_transform(X)
print(emb.shape)  # (100, 2)
```

## Pivot: TruncatedSVD instead of another PCA

PCA is already implemented in `src/pca.py` (`pca_manual`, `pca_svd`, and `variance_retained`). Those routines subtract the column mean before factoring. Centering a bag-of-words or TF-IDF matrix fills in structural zeros and destroys sparsity, so this change does not add a second PCA.

`TruncatedSVD` is the sparse-matrix counterpart. It factors the matrix as stored, accepts SciPy CSR/CSC input, and exposes `fit`, `transform`, `inverse_transform`, and `explained_variance_ratio_`. `n_components` is either an integer rank or a variance fraction such as `0.95`.

Run the example from the repository root with `src` on `PYTHONPATH` (the same layout `run.py` uses).

```python
import numpy as np
from scipy import sparse
from truncated_svd import TruncatedSVD

rng = np.random.default_rng(0)
counts = sparse.csr_matrix(rng.poisson(0.4, size=(80, 30)).astype(float))
document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
idf = np.log((1 + counts.shape[0]) / (1 + document_frequency)) + 1.0
tfidf = counts.multiply(idf)

svd = TruncatedSVD(n_components=0.95)
embedding = svd.fit_transform(tfidf)
restored = svd.inverse_transform(embedding)

print(embedding.shape)  # (80, k)
print(svd.n_components_, round(float(svd.explained_variance_ratio_.sum()), 3))
print(restored.shape)  # (80, 30)
```

An integer rank works the same way: `TruncatedSVD(n_components=10)`. The embedding is dense. `inverse_transform` does not add a mean, because the decomposition never subtracted one.


## Spectral Embedding / Laplacian Eigenmaps

`SpectralEmbedding` in `src/spectral_embedding.py` builds a symmetric k-nearest-neighbour
affinity graph, forms the symmetric normalised Laplacian
``L = I - D^{-1/2} W D^{-1/2}``, and embeds each point with the smallest
non-trivial eigenvectors (Belkin & Niyogi Laplacian Eigenmaps). Use it when
local neighbourhood connectivity matters more than global geodesic distances
(Isomap) or reconstruction weights (LLE).

`n_neighbors` controls the graph; `affinity` is ``"nearest_neighbors"`` (binary)
or ``"rbf"`` (Gaussian heat kernel on the same edges). Classic Laplacian
Eigenmaps is transductive: there is no out-of-sample ``transform``.

```python
import numpy as np
from spectral_embedding import SpectralEmbedding, spectral_reduce

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(loc=-2.0, size=(50, 8)),
    rng.normal(loc=2.0, size=(50, 8)),
])
emb = SpectralEmbedding(n_components=2, n_neighbors=10).fit_transform(X)
print(emb.shape)  # (100, 2)
```


## Diffusion Maps

`DiffusionMaps` in `src/diffusion_maps.py` builds a Gaussian affinity (kNN or
dense RBF), optionally density-normalises it with Coifman–Lafon ``alpha``,
forms the row-stochastic diffusion / Markov matrix, and embeds each point with
``lambda^t * eigenvector`` coordinates (skipping the trivial first mode). Use it
when a diffusion geometry / multi-scale random-walk view of the data matters.

`n_neighbors` / ``epsilon`` control the affinity; ``t`` is diffusion time;
``alpha`` is density normalisation (``0`` skips it). Classic Diffusion Maps is
transductive: there is no out-of-sample ``transform``.

```python
import numpy as np
from diffusion_maps import DiffusionMaps, diffusion_reduce

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(loc=-2.0, size=(50, 8)),
    rng.normal(loc=2.0, size=(50, 8)),
])
emb = DiffusionMaps(n_components=2, n_neighbors=10, t=1.0).fit_transform(X)
print(emb.shape)  # (100, 2)
frame = diffusion_reduce(X, n_components=2)
```

`diffusion_reduce` returns a pandas DataFrame with columns `DM1`, `DM2`, …

## Factor Analysis

`FactorAnalysis` in `src/factor_analysis.py` models each row as a linear mix of
a few latent factors plus independent per-feature noise (uniquenesses). PCA
rotates the leading covariance eigenvectors; FA instead estimates loadings
`components_` and a diagonal `noise_variance_` with EM (Rubin & Thayer style).
`fit_transform` / `transform` return the posterior factor means.

```python
from factor_analysis import FactorAnalysis, factor_analysis_reduce
import numpy as np

X = np.random.default_rng(0).normal(size=(200, 8))
emb = FactorAnalysis(n_components=2, max_iter=200).fit_transform(X)
frame = factor_analysis_reduce(X, n_components=2)
```

`factor_analysis_reduce` returns a pandas DataFrame with columns `FA1`, `FA2`, …

## FastICA

`FastICA` in `src/ica.py` is a symmetric fixed-point ICA (Hyvärinen's algorithm with the logcosh contrast). PCA and truncated SVD factor variance. ICA whitens that subspace and then rotates it so the coordinates are as independent as the contrast can make them. Use it when the observed columns are an unknown linear mix of a few non-Gaussian sources.

`n_components` is the number of sources. `None` keeps every feature. `fit`, `transform`, and `inverse_transform` follow the same estimator shape as `TruncatedSVD`. Recovered sources and mixing columns are only determined up to sign and order.

Run the example from the repository root with `src` on `PYTHONPATH`.

```python
import numpy as np
from ica import FastICA

rng = np.random.default_rng(0)
t = np.linspace(0, 8, 1500, endpoint=False)
sources = np.column_stack(
    [
        np.sin(2 * np.pi * 0.7 * t),
        np.sign(np.sin(2 * np.pi * 1.5 * t)),
        rng.laplace(size=t.shape[0]),
    ]
)
sources -= sources.mean(axis=0)
sources /= sources.std(axis=0)
observed = sources @ rng.normal(size=(6, 3)).T

ica = FastICA(n_components=3, random_state=0)
recovered = ica.fit_transform(observed)
restored = ica.inverse_transform(recovered)

print(recovered.shape)  # (1500, 3)
print(ica.n_iter_, round(float(np.mean((observed - restored) ** 2)), 6))
```

`components_` is the unmixing matrix, shape `(n_components, n_features)`. `mixing_` maps sources back to the centered features. `inverse_transform` adds the training mean.

## NMF

`NMF` in `src/nmf.py` factors a non-negative matrix as `X ≈ W @ H` with Lee–Seung multiplicative updates. `W` is the sample embedding and `H` (`components_`) is the basis. Both stay non-negative, so the parts can be read as topics, spectra, or additive features. PCA and FastICA are free to use negative loadings; this one is not.

The implementation is NumPy only. `nmf_reduce` follows `pca_manual`: it returns the embedding and an info dict. A DataFrame comes back as a DataFrame with columns `NMF1`, `NMF2`, ...; an array comes back as an array. `components`, `reconstruction_error` (mean squared residual), and `n_iter` live in the info dict. The class also exposes `fit`, `transform`, `fit_transform`, and `inverse_transform`.

`n_components` is an integer rank, at most `min(n_samples, n_features)`. The default start is non-negative double SVD (`init="nndsvda"`). `init="random"` uses `random_state`. Negative, NaN, or infinite entries are rejected. The comparison pipeline shifts each synthetic column so its minimum is zero before calling `nmf_reduce`, because those features are signed.

Run the example from the repository root with `src` on `PYTHONPATH`.

```python
import numpy as np
from nmf import NMF, nmf_reduce

rng = np.random.default_rng(0)
topics = rng.random((4, 12))
weights = rng.dirichlet(np.ones(4), size=50)
documents = weights @ topics

embedding, info = nmf_reduce(documents, n_components=4, random_state=0)
model = NMF(n_components=4, random_state=0)
codes = model.fit_transform(documents)
restored = model.inverse_transform(codes)

print(embedding.shape)  # (50, 4)
print(info["components"].shape)  # (4, 12)
print(float(embedding.min()), round(info["reconstruction_error"], 6))
print(restored.shape)  # (50, 12)
```

`codes @ model.components_` is the reconstruction. There is no mean to add.

## Evaluation

- Variance retention
- Reconstruction error
- Silhouette score on reduced embeddings
- Visual separation quality
- Unsupervised neighbourhood-rank quality (`embedding_quality.py`). The
  pipeline records trustworthiness, continuity and `R_NX` AUC for every
  2-D embedding.

## Embedding quality (trustworthiness, continuity, co-ranking)

Silhouette needs labels, and reconstruction error only applies to linear
maps. `src/embedding_quality.py` adds metrics for any embedding that only
compare neighbour ranks in the input and the embedding:

| Function | Measures | Perfect / random |
| --- | --- | --- |
| `trustworthiness(X, Y, k)` | Intrusions: embedding neighbours that were far in the input (Venna & Kaski, 2001) | 1 / ~0.5 |
| `continuity(X, Y, k)` | Extrusions: input neighbours pushed away | 1 / ~0.5 |
| `neighborhood_preservation(X, Y, k)` | `Q_NX(k)`, the mean overlap of the k-NN sets | 1 / `k/(n-1)` |
| `lcmc(X, Y, k)` | `Q_NX(k) - k/(n-1)` (Chen & Buja, 2009) | `1 - k/(n-1)` / 0 |
| `rnx_curve`, `rnx_auc` | Rescaled `R_NX(K)` for all K, and its area on a log-K axis (Lee et al., 2015) | 1 / 0 |
| `coranking_matrix(X, Y)` | `Q[k-1, l-1]`: pairs ranked k-th in the input and l-th in the embedding (Lee & Verleysen, 2009) | diagonal |

`trustworthiness` matches `sklearn.manifold.trustworthiness` exactly
(Euclidean distances, stable index tie-breaking). `embedding_quality(X, Y, k)`
returns all the scalars from one ranking pass.

```python
from embedding_quality import embedding_quality
from isomap import Isomap

Y = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
print(embedding_quality(X, Y, n_neighbors=10))
```


## Classic t-SNE (NumPy)

The sklearn wrapper in `supervised_methods.tsne_reduce` is convenient for
quick plots. `src/tsne.py` adds the classic exact algorithm matching the
style of `FastICA` / `NMF` / `TruncatedSVD`: a `TSNE` class with
`fit` / `fit_transform`, binary-search perplexity affinities, early
exaggeration, and adaptive gains. No sklearn dependency.

```python
import numpy as np
from tsne import TSNE, tsne_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 8))
embedding = TSNE(n_components=2, perplexity=20.0, learning_rate=200.0, n_iter=500, random_state=0).fit_transform(X)
print(embedding.shape)  # (120, 2)
```

## Isomap

`Isomap` in `src/isomap.py` builds a k-nearest-neighbour graph, approximates
geodesic distances with Floyd–Warshall, and embeds them with classical MDS
(Tenenbaum, de Silva & Langford, 2000). Use it when the data lie on a
nonlinear manifold that is locally Euclidean.

```python
import numpy as np
from isomap import Isomap, isomap_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 8))
emb = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
print(emb.shape)  # (120, 2)
```

`n_neighbors` controls the neighbourhood graph. If the raw kNN graph is
disconnected, shortest inter-component edges are bridged automatically.
`isomap_reduce` returns a pandas DataFrame with columns `ISO1`, `ISO2`, …

## LLE (Locally Linear Embedding)

`LLE` in `src/lle.py` reconstructs each point as a linear combination of its
k nearest neighbours, then finds a low-dimensional embedding that preserves
those reconstruction weights (Roweis & Saul, 2000). Use it when the data lie
on a nonlinear manifold that is locally linear.

```python
import numpy as np
from lle import LLE, lle_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 8))
emb = LLE(n_components=2, n_neighbors=10).fit_transform(X)
print(emb.shape)  # (120, 2)
```

`n_neighbors` controls the local patches. `lle_reduce` returns a pandas
DataFrame with columns `LLE1`, `LLE2`, …

## Classical MDS

`ClassicalMDS` in `src/mds.py` embeds points from pairwise Euclidean
distances via double-centering and the leading eigenvectors of the
resulting Gram matrix (Torgerson scaling). Pass `metric="precomputed"` to
supply a square distance matrix directly.

```python
import numpy as np
from mds import ClassicalMDS, mds_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 6))
emb = ClassicalMDS(n_components=2).fit_transform(X)
print(emb.shape)  # (80, 2)
```

`mds_reduce` returns a pandas DataFrame with columns `MDS1`, `MDS2`, …


## Sammon Mapping

`Sammon` in `src/sammon.py` is Sammon's 1969 nonlinear mapping: it iteratively
minimises a **fractional** stress that weights relative distance errors more
heavily for nearby pairs than classical MDS does. Initialisation defaults to
classical MDS; updates use Sammon's steepest-descent step with a magic factor
(default `0.3`). Classic Sammon mapping is transductive — there is no
out-of-sample `transform`.

```python
import numpy as np
from sammon import Sammon, sammon_reduce

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(loc=-2.0, size=(40, 6)),
    rng.normal(loc=2.0, size=(40, 6)),
])
emb = Sammon(n_components=2, max_iter=100).fit_transform(X)
print(emb.shape)  # (80, 2)
print(round(Sammon(n_components=2).fit(X).stress_, 6))

frame = sammon_reduce(X, n_components=2)
```

`sammon_reduce` returns a pandas DataFrame with columns `SAM1`, `SAM2`, …


## Kernel PCA

`KernelPCA` in `src/kernel_pca.py` embeds points via a centred Gram matrix
and its leading eigenvectors (Schölkopf, Smola & Müller, 1998). Supported
kernels are `rbf`, `linear`, and `poly`. `transform` provides the
Nyström-style out-of-sample extension.

```python
import numpy as np
from kernel_pca import KernelPCA, kernel_pca_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 6))
emb = KernelPCA(n_components=2, kernel="rbf").fit_transform(X)
print(emb.shape)  # (80, 2)
```

`kernel_pca_reduce` returns a pandas DataFrame with columns `KPCA1`, `KPCA2`, …
