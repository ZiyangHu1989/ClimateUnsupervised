# Comprehensive evaluation of global climate clusters

## Methodology

The frozen K-means and GMM assignments were evaluated in their common saved PCA representation. Köppen–Geiger labels were loaded only after frozen-input verification and were never used for model selection. Silhouette, Davies–Bouldin, and Calinski–Harabasz use the same reproducible area-proportional sample.

## Dataset and spatial alignment

The processed dataset contains 94,206 land cells. Köppen sampling produced 85,858 valid matches (91.1%). The reference is EPSG:4326 at 0.008333° resolution. Reference confidence available: False.

## RQ1 — Internal coherence and reproducibility

| Method | Silhouette | Davies–Bouldin | Calinski–Harabasz | Seed ARI mean±SD | Bootstrap ARI mean±SD | Spatial ARI mean±SD |
|---|---:|---:|---:|---:|---:|---:|
| KMEANS | 0.379 | 0.900 | 6850.8 | 0.991±0.007 | 0.724±0.328 | 0.904±0.068 |
| GMM | 0.361 | 0.948 | 5656.5 | 0.729±0.202 | 0.656±0.140 | 0.662±0.128 |

For GMM, 7.0% of cells have maximum posterior below 0.8. Normalized entropy is reported in `evaluation-summary.json`.

## RQ2 — Agreement with Köppen–Geiger

| Method | Level | ARI | NMI / V-measure | Area ARI | Area NMI / V-measure |
|---|---|---:|---:|---:|---:|
| KMEANS | primary_groups_5 | 0.544 | 0.600 | 0.450 | 0.536 |
| KMEANS | subclasses_30 | 0.472 | 0.552 | 0.261 | 0.475 |
| GMM | primary_groups_5 | 0.510 | 0.544 | 0.350 | 0.462 |
| GMM | subclasses_30 | 0.470 | 0.535 | 0.224 | 0.463 |

NMI uses arithmetic-mean normalization; with this convention it equals V-measure, so the two are not independent signals. Full contingency, Jaccard, precision, recall, and F1 matrices are stored in `class-overlap.csv`.

## RQ3 — Geographic agreement and disagreement

Regional correspondence is computed after mapping each learned cluster to its strongest Jaccard-overlap class; raw IDs are never compared. The named geographic windows are diagnostics rather than exact topographic or biome masks. See `regional-correspondence.csv` and the maps.

## RQ4 — Physical interpretation and subdivisions

Monthly physical-variable profiles cover 10 learned clusters and include weighted dispersion and quantile bands.

| Method | Cluster | Area | Mean temperature | Temperature range | Annual precipitation |
|---|---:|---:|---:|---:|---:|
| KMEANS | 0 | 36.7% | 20.8 °C | 3.2 °C | 1623 mm |
| KMEANS | 1 | 32.5% | -1.9 °C | 30.5 °C | 618 mm |
| KMEANS | 2 | 22.6% | 22.8 °C | 9.7 °C | 248 mm |
| KMEANS | 3 | 8.1% | -35.1 °C | 18.5 °C | 126 mm |
| GMM | 0 | 35.3% | -0.0 °C | 31.3 °C | 677 mm |
| GMM | 1 | 17.7% | 21.8 °C | 0.9 °C | 2169 mm |
| GMM | 2 | 17.6% | 20.8 °C | 2.7 °C | 656 mm |
| GMM | 3 | 10.4% | 21.5 °C | 21.4 °C | 57 mm |
| GMM | 4 | 10.0% | 25.2 °C | 5.4 °C | 1004 mm |
| GMM | 5 | 9.1% | -32.2 °C | 20.2 °C | 218 mm |

The analysis contains 31 reference-class/cluster subdivision comparisons in which the secondary cluster covers at least 10% of reference-class area. The largest physical contrasts are listed below as diagnostics, not significance tests.

| Method | Level | Köppen ID | Clusters | Δ temperature | Δ annual precipitation |
|---|---|---:|---|---:|---:|
| GMM | primary_groups_5 | 2 | 1 vs 0 | 21.8 °C | 1492 mm |
| GMM | subclasses_30 | 14 | 1 vs 0 | 21.8 °C | 1492 mm |
| GMM | subclasses_30 | 15 | 1 vs 0 | 21.8 °C | 1492 mm |
| GMM | subclasses_30 | 16 | 1 vs 0 | 21.8 °C | 1492 mm |
| KMEANS | primary_groups_5 | 4 | 3 vs 1 | 33.3 °C | 491 mm |
| KMEANS | subclasses_30 | 30 | 3 vs 1 | 33.3 °C | 491 mm |
| KMEANS | subclasses_30 | 7 | 1 vs 0 | 22.7 °C | 1005 mm |
| KMEANS | subclasses_30 | 18 | 1 vs 0 | 22.7 °C | 1005 mm |

### Regional correspondence for primary groups

| Method | Diagnostic region | Area-weighted correspondence |
|---|---|---:|
| KMEANS | Andes window | 62.0% |
| KMEANS | Himalaya-Tibetan window | 33.7% |
| KMEANS | Sahara-Sahel window | 91.7% |
| KMEANS | South Asian monsoon window | 52.5% |
| KMEANS | North American transition window | 62.6% |
| KMEANS | Eurasian transition window | 58.9% |
| KMEANS | Polar latitudes | 80.3% |
| GMM | Andes window | 74.0% |
| GMM | Himalaya-Tibetan window | 30.2% |
| GMM | Sahara-Sahel window | 76.5% |
| GMM | South Asian monsoon window | 44.2% |
| GMM | North American transition window | 73.6% |
| GMM | Eurasian transition window | 65.1% |
| GMM | Polar latitudes | 81.7% |

## Limitations and future work

- Köppen is a development benchmark, not an untouched confirmatory test.
- Spatial block stability exposes geographic sampling sensitivity that ordinary seed repeats may understate.
- The Köppen raster has no confidence band, so reference-confidence analysis cannot be performed.
- Named regional windows do not replace elevation, terrain, or formal biome masks.
- Temporal stability and a genuinely unseen external reference remain future confirmatory work.

## Reproduction

```bash
climate-unsupervised evaluate-full
```

Generated 15 evaluation figures under `results/figures/evaluation/`.
