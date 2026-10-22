# ClimateUnsupervised

An area-aware, reproducible prototype for discovering global climate regions
without using geographic coordinates or predefined climate labels as model
features. It builds 24-dimensional 1991–2020 monthly climatologies from
ERA5-Land, compares PCA-based K-means and Gaussian mixture models, freezes all
unsupervised choices, and only then evaluates against Köppen–Geiger classes.

## Method

Each retained 0.5° land cell has twelve monthly mean temperatures in °C and
twelve monthly precipitation totals in mm/month. The model applies `log1p` to
precipitation, standardizes with exact cell-area weights, and retains enough
PCA components to explain at least 99% of variance. PCA and GMM use
area-proportional bootstrap samples because those estimators do not accept
sample weights. Latitude and longitude are metadata only; cosine-latitude
weights correct for unequal grid-cell area.

K-means is fitted with area weights. GMM candidates compare diagonal and full
covariance on a fixed area-proportional bootstrap and are then applied to all
cells. Both algorithms evaluate `K=2,...,15`; stability is measured across
independent area-weighted bootstrap samples rather than initialization alone.
Candidates must pass stability and minimum-cluster-area safeguards. GMM is
selected by silhouette after those safeguards, while BIC/AIC remain density-fit
diagnostics. Köppen–Geiger data is inaccessible to model selection and is read
only after a frozen artifact manifest has been created.

External evaluation is deliberately descriptive. It reports both the original
30 Köppen subclasses and the fixed five primary groups (A--E), including ARI,
AMI/NMI, homogeneity, completeness, V-measure, and area-weighted counterparts.
No external metric is used to choose PCA dimension, covariance type, or K.
The development/confirmation boundary and anti-reward-hacking rules are recorded
in [`research_protocol.md`](research_protocol.md).

## Setup

Using Conda (recommended for the geospatial stack):

```bash
conda create -p .conda-env python=3.11 pip -y
conda run -p .conda-env python -m pip install -e '.[dev,grib]'
```

Alternatively, create a Python virtual environment and install the same extras.

Create a Copernicus Climate Data Store account, accept the ERA5-Land dataset
terms, and configure `~/.cdsapirc` as described by the CDS. The default
Köppen–Geiger URL points to the 1991–2020 GeoTIFF distributed by Stanford's
Natural Capital Data Hub.

## Run

Run individual resumable stages:

```bash
climate-unsupervised download-era5
climate-unsupervised preprocess
climate-unsupervised fit
climate-unsupervised visualize
climate-unsupervised freeze
climate-unsupervised download-koppen
climate-unsupervised evaluate
climate-unsupervised visualize
climate-unsupervised evaluate-full
```

Or run the complete workflow:

```bash
climate-unsupervised all
```

### Australia-only run

The repository also includes an isolated regional configuration for mainland
Australia and Tasmania. It reuses the completed global ERA5-Land feature file,
filters cell centers with a pinned Natural Earth country boundary, and writes
all derived artifacts beneath `runs/australia/` so the global run is preserved.
New Zealand, Indonesia, Papua New Guinea, and smaller offshore components are
excluded.

```bash
climate-unsupervised regional-all --config configs/australia.yaml
```

The regional workflow fits PCA, K-means, and GMM only on Australian cells,
freezes the selected models, evaluates against the Australian part of the
Köppen–Geiger raster, and creates an Australia-specific report. Run
`prepare-region` by itself to rebuild only the regional feature subset.

The initial maximum-silhouette rule produced an overly coarse two-cluster
Australian baseline. The multiscale workflow instead searches K=3 through 12
with K-means, diagonal/full GMM, BIRCH, and Ward clustering, then reports every
resolution supported by internal stability and separation safeguards:

```bash
climate-unsupervised multiscale-all --config configs/australia-multiscale.yaml
```

Köppen labels are read only after the internally supported resolutions and
representative algorithms have been frozen.

Every command accepts `--config`; destructive recreation requires `--force`.
The `freeze` command refuses to overwrite a frozen manifest without that flag.
The evaluation stage verifies that every frozen input still matches its stored
SHA-256 digest.

## Storage

The final 24-feature dataset is expected to occupy roughly 7–10 MB. Processed
artifacts and figures should remain below 200 MB. Acquisition is performed in
yearly chunks: each native year is aggregated to 0.5° and can then be deleted,
keeping expected peak workspace usage near 1–3 GB rather than retaining roughly
19 GB of uncompressed native 0.1° values.

By default, `download-era5` immediately aggregates and validates each yearly
file before deleting that native-resolution chunk and advancing to the next
year. Three yearly requests run concurrently by default; adjust
`era5.download_workers` in the configuration if CDS throttles the account or
local memory is constrained. Downloaded data, binary models, and figures are excluded from Git. Source,
configuration, tests, documentation, and compact JSON metrics/manifests can be
tracked; manifests store project-relative paths and content hashes.

## Outputs

- `data/processed/climate_features.npz`: features and non-feature metadata.
- `results/models/`: scaler, PCA, selected models, and assignments.
- `results/metrics/internal.json`: candidate selection evidence.
- `results/manifests/frozen-model.json`: immutable pre-evaluation boundary.
- `results/metrics/external.json`: weighted and unweighted external metrics.
- `results/figures/`: PCA, model-selection, profile, global, and disagreement plots.
- `results/evaluation-report.md`: comprehensive five-stage evaluation report.
- `results/metrics/evaluation-summary.json`: machine-readable evaluation index.
- `results/figures/evaluation/`: comprehensive evaluation figures.
- `runs/australia/`: isolated Australian data, models, metrics, figures, manifests, and report.
- `runs/australia-multiscale/`: multi-algorithm Australian benchmark and supported-resolution report.

## Australian climate feature expansion

The existing-data feature experiment uses the 30 yearly monthly files to derive
110 named columns across four representations: the original monthly baseline,
separated climate level/seasonal profiles, physical summaries, and summaries
plus variability/persistence. It compares direct geometry with area-weighted
99%-variance PCA using the same five algorithm variants and K=3–12.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  climate-unsupervised feature-all --config configs/australia-features.yaml
```

`feature-prepare` builds the regional time series and feature registry;
`feature-fit` runs restartable comparisons, family ablations, and freezes results;
`feature-evaluate` renders the report and evaluates the frozen labels against
the local Köppen raster. No new climate data are downloaded. `--force` explicitly
recomputes the stages; otherwise incompatible checkpoints require a new run
directory or explicit recomputation.

Preprocessing is refitted within each bootstrap/spatial sample. Five-year
temporal block resamples recompute features and exclude artificial time joins.
Correlation pruning and equal family-variance weighting apply to engineered
representations; the refreshed monthly baseline provides a controlled comparator.

- [Research feature catalogue](docs/climate-feature-catalogue.md)
- [Targeted new-data acquisition specification](docs/climate-data-acquisition.md)
- [Experiment report](runs/australia-features/results/feature-expansion-report.md)
- [Exact feature metadata](runs/australia-features/results/metrics/feature-catalogue.json)

## Testing

```bash
pytest
```

Tests use synthetic climate regimes and a synthetic categorical raster; they do
not require network access or CDS credentials.
