# Comprehensive evaluation of Australian climate clusters

## Methodology

The frozen K-means and GMM assignments were evaluated in their common saved PCA representation. Köppen–Geiger labels were loaded only after frozen-input verification and were never used for model selection. Silhouette, Davies–Bouldin, and Calinski–Harabasz use the same reproducible area-proportional sample.

## Dataset and spatial alignment

The processed dataset contains 2,770 land cells. Köppen sampling produced 2,753 valid matches (99.4%). The reference is EPSG:4326 at 0.008333° resolution. Reference confidence available: False.

## RQ1 — Internal coherence and reproducibility

| Method | Silhouette | Davies–Bouldin | Calinski–Harabasz | Seed ARI mean±SD | Bootstrap ARI mean±SD | Spatial ARI mean±SD |
|---|---:|---:|---:|---:|---:|---:|
| KMEANS | 0.385 | 1.002 | 2314.5 | 0.994±0.007 | 0.930±0.051 | 0.893±0.064 |
| GMM | 0.358 | 0.946 | 1561.6 | 0.994±0.006 | 0.972±0.017 | 0.958±0.019 |

For GMM, 4.8% of cells have maximum posterior below 0.8. Normalized entropy is reported in `evaluation-summary.json`.

For regional interpretation, GMM is the more robust saved partition because its bootstrap and spatial-block stability are higher. K-means remains the stronger silhouette baseline. This comparison uses internal diagnostics only; Köppen agreement did not determine the recommendation.

## RQ2 — Agreement with Köppen–Geiger

| Method | Level | ARI | NMI / V-measure | Area ARI | Area NMI / V-measure |
|---|---|---:|---:|---:|---:|
| KMEANS | primary_groups_5 | 0.034 | 0.172 | 0.034 | 0.171 |
| KMEANS | subclasses_30 | 0.041 | 0.175 | 0.044 | 0.174 |
| GMM | primary_groups_5 | 0.177 | 0.216 | 0.182 | 0.218 |
| GMM | subclasses_30 | 0.151 | 0.196 | 0.161 | 0.201 |

NMI uses arithmetic-mean normalization; with this convention it equals V-measure, so the two are not independent signals. Full contingency, Jaccard, precision, recall, and F1 matrices are stored in `class-overlap.csv`.

## RQ3 — Geographic agreement and disagreement

Geographic maps show assignments and Köppen correspondence across Australia. No informal subregional windows are used in this regional analysis.

## RQ4 — Physical interpretation and subdivisions

Monthly physical-variable profiles cover 4 learned clusters and include weighted dispersion and quantile bands.

| Method | Cluster | Area | Mean temperature | Temperature range | Annual precipitation |
|---|---:|---:|---:|---:|---:|
| KMEANS | 0 | 51.8% | 25.5 °C | 12.4 °C | 490 mm |
| KMEANS | 1 | 48.2% | 18.9 °C | 14.7 °C | 410 mm |
| GMM | 0 | 74.1% | 20.7 °C | 15.1 °C | 377 mm |
| GMM | 1 | 25.9% | 26.8 °C | 10.1 °C | 663 mm |

The analysis contains 7 reference-class/cluster subdivision comparisons in which the secondary cluster covers at least 10% of reference-class area. The largest physical contrasts are listed below as diagnostics, not significance tests.

| Method | Level | Köppen ID | Clusters | Δ temperature | Δ annual precipitation |
|---|---|---:|---|---:|---:|
| GMM | primary_groups_5 | 1 | 0 vs 1 | 6.1 °C | 286 mm |
| GMM | subclasses_30 | 4 | 0 vs 1 | 6.1 °C | 286 mm |
| GMM | subclasses_30 | 6 | 0 vs 1 | 6.1 °C | 286 mm |
| KMEANS | primary_groups_5 | 1 | 0 vs 1 | 6.6 °C | 80 mm |
| KMEANS | subclasses_30 | 4 | 0 vs 1 | 6.6 °C | 80 mm |
| KMEANS | subclasses_30 | 6 | 0 vs 1 | 6.6 °C | 80 mm |
| KMEANS | subclasses_30 | 11 | 1 vs 0 | 6.6 °C | 80 mm |

## Limitations and future work

- Köppen is a development benchmark, not an untouched confirmatory test.
- Spatial block stability exposes geographic sampling sensitivity that ordinary seed repeats may understate.
- The Köppen raster has no confidence band, so reference-confidence analysis cannot be performed.
- Temporal stability and a genuinely unseen external reference remain future confirmatory work.

## Reproduction

```bash
climate-unsupervised evaluate-full --config configs/australia.yaml
```

Generated 15 evaluation figures for this run.
