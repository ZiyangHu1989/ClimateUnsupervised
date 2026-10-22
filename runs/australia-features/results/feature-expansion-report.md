# Australian climate feature-expansion experiment

Analysed 2,770 mainland Australian and Tasmanian cells at 0.5°, 1991–2020. Derived 110 named candidate columns, including monthly baseline and profile representations.

## Main findings

The refreshed monthly baseline supports a displayed K=6 without PCA, while its PCA version displays K=4. Thus the earlier general conclusion that all K≥5 solutions are unstable was too strong: representation and fitting/validation choices matter. This comparison does not isolate PCA as the sole cause, because it also changes the fitted geometry and the silhouette reference used for selection.

Physical summaries support K=5, with temporal ARI 0.803; compare this with the configured temporal threshold of 0.80. Their displayed Köppen subclass ARI is 0.398. Adding variability/persistence instead displays K=4 with subclass ARI 0.331. More feature columns have not established a clearly superior detailed national classification. External correspondence alone is not a quality verdict.

The maps still show broad regimes. A 30-year monthly temperature/rainfall record does not contain daily heat, frost, rain-event structure or direct atmospheric moisture-demand information. The next data stage is therefore scientifically motivated by missing measurements, rather than a promise of a larger K.

[Broader feature catalogue](../../../docs/climate-feature-catalogue.md) · [All implemented definitions](../../../docs/implemented-climate-features.md) · [Targeted data acquisition specification](../../../docs/climate-data-acquisition.md)

## Comparison

Each representation was tested with five algorithm variants and K=3–12. Displayed results use the largest K passing the prior internal shortlist rules AND temporal ARI ≥0.80. Where none passes, the display is explicitly diagnostic, chosen by its weakest stability dimension. All resolutions and failures remain in the JSON audit. These displays are not a single winning national classification.

| Representation | Features → dimensions | Method / K | Bootstrap | Spatial | Temporal | Min area | Köppen subclass ARI | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| baseline-direct | 24 → 24 | kmeans / 6 | 0.894 | 0.921 | 0.886 | 6.9% | 0.354 | temporally_confirmed |
| baseline-pca | 24 → 8 | kmeans / 4 | 0.905 | 0.954 | 0.905 | 13.7% | 0.314 | temporally_confirmed |
| profiles-direct | 22 → 22 | kmeans / 4 | 0.882 | 0.917 | 0.877 | 17.7% | 0.394 | temporally_confirmed |
| profiles-pca | 22 → 10 | kmeans / 4 | 0.886 | 0.919 | 0.878 | 17.5% | 0.399 | temporally_confirmed |
| summaries-direct | 23 → 23 | kmeans / 5 | 0.870 | 0.957 | 0.803 | 12.9% | 0.398 | temporally_confirmed |
| summaries-pca | 23 → 11 | kmeans / 5 | 0.906 | 0.865 | 0.802 | 12.8% | 0.399 | temporally_confirmed |
| variability-direct | 42 → 42 | kmeans / 4 | 0.929 | 0.940 | 0.836 | 16.9% | 0.331 | temporally_confirmed |
| variability-pca | 42 → 21 | kmeans / 4 | 0.911 | 0.937 | 0.847 | 16.9% | 0.330 | temporally_confirmed |

![Classification comparison](figures/features/classification-comparison.png)

## Selection and area sensitivity

The old silhouette-retention and area rules are applied separately inside each representation. Silhouettes across different feature geometries are not directly comparable. The 0% area sensitivity still requires the requested number of nonempty clusters; it is not endorsement of arbitrarily tiny classes.

| Representation | Prior-rule shortlist | Temporally confirmed | Confirmed at 0% / 0.5% / 1% / 2% area |
|---|---|---|---|
| baseline-direct | 3, 4, 6 | 3, 4, 6 | 3, 4, 6 / 3, 4, 6 / 3, 4, 6 / 3, 4, 6 |
| baseline-pca | 3, 4 | 4 | 4 / 4 / 4 / 4 |
| profiles-direct | 3, 4 | 3, 4 | 3, 4 / 3, 4 / 3, 4 / 3, 4 |
| profiles-pca | 3, 4 | 3, 4 | 3, 4 / 3, 4 / 3, 4 / 3, 4 |
| summaries-direct | 3, 4, 5 | 3, 4, 5 | 3, 4, 5 / 3, 4, 5 / 3, 4, 5 / 3, 4, 5 |
| summaries-pca | 3, 4, 5 | 3, 4, 5 | 3, 4, 5 / 3, 4, 5 / 3, 4, 5 / 3, 4, 5 |
| variability-direct | 3, 4 | 3, 4 | 3, 4 / 3, 4 / 3, 4 / 3, 4 |
| variability-pca | 3, 4 | 3, 4 | 3, 4 / 3, 4 / 3, 4 / 3, 4 |

## Feature-family ablations

Each ablation repeats all three stability checks at its full representation's displayed method/K. Low agreement means the family changes the partition; it does not prove that the change improves climate classification. Diagnostic references remain diagnostic.

| Representation | Omitted family | ARI to full | Bootstrap | Spatial | Temporal |
|---|---|---:|---:|---:|---:|
| summaries-direct | coupling | 0.603 | 0.814 | 0.771 | 0.606 |
| summaries-direct | rainfall | 0.715 | 0.863 | 0.777 | 0.670 |
| summaries-direct | rainfall_seasonality | 0.830 | 0.896 | 0.895 | 0.854 |
| summaries-direct | temperature | 0.877 | 0.937 | 0.959 | 0.799 |
| summaries-direct | thermal_seasonality | 0.836 | 0.928 | 0.963 | 0.717 |
| summaries-pca | coupling | 0.603 | 0.809 | 0.786 | 0.600 |
| summaries-pca | rainfall | 0.718 | 0.748 | 0.898 | 0.671 |
| summaries-pca | rainfall_seasonality | 0.833 | 0.891 | 0.887 | 0.857 |
| summaries-pca | temperature | 0.880 | 0.872 | 0.959 | 0.810 |
| summaries-pca | thermal_seasonality | 0.836 | 0.924 | 0.896 | 0.725 |
| variability-direct | coupling | 0.864 | 0.933 | 0.942 | 0.818 |
| variability-direct | persistence | 0.921 | 0.947 | 0.942 | 0.851 |
| variability-direct | rainfall | 0.900 | 0.868 | 0.744 | 0.775 |
| variability-direct | rainfall_seasonality | 0.881 | 0.825 | 0.710 | 0.832 |
| variability-direct | temperature | 0.889 | 0.949 | 0.933 | 0.800 |
| variability-direct | thermal_seasonality | 0.854 | 0.854 | 0.955 | 0.822 |
| variability-direct | variability | 0.857 | 0.911 | 0.939 | 0.843 |
| variability-pca | coupling | 0.864 | 0.922 | 0.938 | 0.820 |
| variability-pca | persistence | 0.904 | 0.944 | 0.941 | 0.853 |
| variability-pca | rainfall | 0.901 | 0.862 | 0.848 | 0.804 |
| variability-pca | rainfall_seasonality | 0.843 | 0.805 | 0.745 | 0.829 |
| variability-pca | temperature | 0.889 | 0.941 | 0.934 | 0.805 |
| variability-pca | thermal_seasonality | 0.849 | 0.882 | 0.953 | 0.829 |
| variability-pca | variability | 0.856 | 0.908 | 0.926 | 0.826 |

## Reliability and interpretation

Scaling, missing-value imputation, within-family correlation pruning, family weighting and PCA are fitted anew in each cell or spatial training sample. Temporal trials recompute physical features from five-year moving blocks and refit preprocessing. Artificial temporal joins are excluded from seasonal windows and persistence calculations. Boundary-censored dry-run estimates remain exploratory.

The benchmark uses ten cell-bootstrap trials, five spatial folds, and five temporal resamples. Non-KMeans fits use at most 2,000 area-resampled training observations. Silhouette uses a reproducible uniform subset of at most 1,500 cells; reported ARIs are unweighted cell agreement, while cluster area, scaling and centroid-label matching use geographic area weights.

Baseline runs retain all variable monthly columns and omit family normalization/pruning, preserving the old feature geometry. PCA here uses exact area-weighted covariance; the previous multiscale code used unweighted PCA. Stability fits use three initializations instead of one, so differences from the old run cannot be attributed solely to added features. Use the refreshed baseline for controlled comparisons.

Derived climatological summaries re-express existing information. Variability and persistence recover information discarded by the old 30-year averaging step. Small seasonal contrasts can still be noise; the temporal tests assess that risk. Local dry-month proxies are not standardized drought indices, and monthly mean temperatures do not measure daily heat or frost extremes.

Candidate failures: 0. Candidates with captured fitting warnings: 0. All messages are saved alongside their candidate metrics. Pairwise stability means are descriptive; their dependent ARI pairs are not independent confidence-interval samples.

Köppen comparison used 2,753/2,770 valid cells only after selection and feature artifacts were frozen. Agreement is descriptive, not an optimization target. A quantitative Bureau of Meteorology comparison was not performed because a matching local reference raster is unavailable; no new climate-data downloads were made.

## Outputs

- [Physical feature maps](figures/features/physical-feature-maps.png)
- [Assignment reproducibility](figures/features/assignment-agreement.png)
- [All stability curves](figures/features/stability-diagnostics.png)
- [Physical profiles in original units](metrics/physical-profiles.csv)
- [Complete experiment and cross-representation comparisons](metrics/feature-experiment.json)
- [Named feature definitions](metrics/feature-catalogue.json)

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 climate-unsupervised feature-all --config configs/australia-features.yaml
```

Checkpoints are reused only if the input series, source code and configuration hashes match. Run `feature-evaluate` to regenerate figures from the frozen results without refitting.
