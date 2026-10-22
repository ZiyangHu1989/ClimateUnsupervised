# Multiscale Australian climate classification

## Why the earlier run produced two clusters

The earlier rule maximized silhouette among eligible candidates. That criterion rewarded the strongest coarse north–south split and did not require useful classification detail. This benchmark treats K=2 only as a macro diagnostic and searches K=3–12 without Köppen labels.

## Internally supported resolutions

| K | Representative | Silhouette | Bootstrap ARI | Spatial ARI | Smallest area | Cross-algorithm ARI |
|---:|---|---:|---:|---:|---:|---:|
| 3 | gmm_diag | 0.317 | 0.884 | 0.903 | 28.2% | 0.585 |
| 4 | kmeans | 0.330 | 0.900 | 0.967 | 13.7% | n/a |

No single K is declared authoritative. A resolution is listed when it passes the common stability, area, and silhouette-retention safeguards and is supported either by cross-algorithm agreement or by independently strong bootstrap and spatial replication within one family.

## Descriptive Köppen comparison

Köppen labels were sampled only after the shortlist was frozen. 2,753 of 2,770 Australian cells had valid reference labels.

| K | Primary-group ARI | Area ARI | Subclass ARI | Area subclass ARI |
|---:|---:|---:|---:|---:|
| 3 | 0.159 | 0.157 | 0.449 | 0.446 |
| 4 | 0.185 | 0.184 | 0.314 | 0.316 |

These external scores describe correspondence and were not used to construct the shortlist.

## Physical profiles

### K=3

| Cluster | Area | Mean temperature | Temperature range | Annual precipitation |
|---:|---:|---:|---:|---:|
| 0 | 43.5% | 22.2 °C | 16.9 °C | 253 mm |
| 1 | 28.3% | 18.1 °C | 12.5 °C | 567 mm |
| 2 | 28.2% | 26.7 °C | 10.3 °C | 641 mm |

### K=4

| Cluster | Area | Mean temperature | Temperature range | Annual precipitation |
|---:|---:|---:|---:|---:|
| 0 | 36.5% | 20.4 °C | 15.6 °C | 320 mm |
| 1 | 29.9% | 24.9 °C | 15.7 °C | 296 mm |
| 2 | 19.9% | 26.7 °C | 8.9 °C | 789 mm |
| 3 | 13.7% | 15.3 °C | 12.5 °C | 645 mm |

## Reproduction

```bash
climate-unsupervised multiscale-all --config configs/australia-multiscale.yaml
```
