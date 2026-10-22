# Research protocol and anti-reward-hacking policy

## Scientific objective

Discover stable, interpretable global climate regions from monthly temperature
and precipitation alone. Agreement with Köppen–Geiger is descriptive evidence,
not the optimization objective: a data-driven partition may validly merge or
split rule-based classes.

## Information boundary

Model selection may use only:

- area-weighted feature geometry;
- silhouette and Davies–Bouldin indices;
- stability across independent area-weighted bootstrap samples;
- minimum geographic-area safeguards;
- GMM likelihood, AIC, and BIC as diagnostics.

It may not use Köppen labels, external ARI/AMI/NMI, disagreement maps, or
cluster-to-Köppen contingency tables. The selected artifacts and protocol are
hashed by `freeze` before `evaluate` reads the external raster.

The baseline Köppen results were already inspected before this revised protocol
was designed. Consequently, all current Köppen results must be described as a
development benchmark, not as an untouched confirmatory test. Further model
changes must be justified by internal diagnostics or scientific assumptions,
not by whether the external score rises. A future confirmatory claim requires a
new preregistered reference or a genuinely unseen spatial/temporal holdout.

## Fixed selection rules

1. Apply `log1p` to precipitation and exact cell-area weights to scaling.
2. Fit PCA and GMM on area-proportional samples drawn with replacement.
3. Retain PCA components to the configured variance threshold (currently 99%).
4. Evaluate K from 2 through 15.
5. Reject candidates below the configured bootstrap-stability or minimum-area
   thresholds.
6. Select K-means and GMM by maximum silhouette among eligible candidates,
   breaking ties by lower Davies–Bouldin index and then lower K.
7. Treat AIC/BIC as density-fit diagnostics, not automatic region-count
   selectors when they decrease throughout the tested K range.
8. Freeze selected models before loading Köppen–Geiger.

Thresholds and tie-breaking rules must not be changed after external evaluation
unless the resulting run is explicitly labeled exploratory.

## External reporting

Always report both levels:

- all available Köppen subclasses (up to 30), showing fine-grained structure;
- the fixed five primary groups A–E, showing macroclimate structure.

Report ARI, AMI/NMI, homogeneity, completeness, and V-measure. Include
area-weighted results because a regular latitude–longitude grid overrepresents
high latitudes. Do not collapse these metrics into a single reward or choose a
model from the external table.

## Remaining robustness work

- Use the implemented geographic-block holdout diagnostic in every final run;
  it is reported separately and does not retroactively change model selection.
- Quantify PCA-dimension sensitivity without selecting dimensions from external
  scores.
- Report confidence/calibration diagnostics for probabilistic assignments.
- Test temporal robustness on non-overlapping climatology periods.
- Reserve a new external reference or preregistered holdout for confirmation.

## Australia-only regional run

The Australia configuration was fixed before its Köppen evaluation. It retains
mainland Australia and Tasmania from the two largest components of the pinned
Natural Earth 1:50m Australia geometry and excludes other countries and smaller
offshore components. It evaluates K from 2 through 10, requires bootstrap ARI
of at least 0.75 and a minimum cluster area of 2%, and uses ten fixed seeds plus
five 2-degree spatial-block folds. PCA retains at least 99% of variance.

The Australian models are fitted only to the regional subset and are frozen
before the Köppen raster is sampled. Their outputs are kept under
`runs/australia/`; the earlier global run remains a separate development
benchmark.

### Multiscale correction

The first Australian run selected K=2 because maximum silhouette rewards the
strongest coarse split. That result is retained as a failed baseline rather
than presented as an adequate climate classification.

The replacement benchmark searches K=3 through 12 using K-means, diagonal and
full-covariance GMM, BIRCH, and Ward clustering. Common safeguards require a
mean bootstrap ARI of at least 0.80, minimum cluster area of 2%, and silhouette
of at least 80% of the best detailed candidate. A resolution must then have
either cross-family ARI of at least 0.55 or a representative solution with
bootstrap ARI of at least 0.85 and spatial-block ARI of at least 0.80. The
single-family route was fixed after an internal-only benchmark showed that
BIRCH and Ward were unsuitable stability comparators for this continuous
climate field, and before the new shortlist was evaluated against Köppen.

All resolutions passing these rules are reported. K=2 remains a macro
diagnostic, no single detailed K is declared authoritative, and Köppen metrics
do not influence the shortlist.

### Feature-expansion experiment (2026-09-20)

The next experiment broadens climate representation before adding data. It keeps
mainland Australia and Tasmania, 1991–2020, and the existing 0.5° cell centres.
The physical feature registry is specified in `docs/climate-feature-catalogue.md`;
future monthly/daily acquisition is specified separately and is not executed
by this experiment.

Compare baseline monthly means, separated level/seasonal profiles, physical
summaries, and summaries plus interannual variability/persistence. Fit each
representation directly and with 99%-variance PCA. Engineered families receive
equal total training variance after weighted standardization and deterministic
within-family correlation pruning at |r|>0.95. Undefined quantities are imputed
using training means; constant/all-missing features are dropped and audited.
Baseline monthly geometry omits pruning and family reweighting.

Refit all preprocessing within 10 area-bootstrap samples and five held-out
spatial folds. Also recompute features in five independent five-year moving-block
year resamples, excluding artificial joins from windows and persistence. Fit
clustering with three initializations in stability trials and ten in final
fits. This reduces, but does not eliminate, optimization noise. PCA now uses
exact area-weighted covariance, so the refreshed baseline—not the historical
run—is the controlled feature comparison.

Retain five methods and K=3–12. Apply the existing silhouette/stability/area rules
within each representation and separately flag temporal mean pairwise ARI≥0.80.
Report area-threshold sensitivity at 0%, 0.5%, 1%, and 2%. Silhouettes across
different feature geometries are not interpreted as directly comparable scores.
Display the largest temporally confirmed shortlisted K per representation;
if none qualifies, show the strongest minimum-of-three-stabilities result as
diagnostic only. Preserve all candidates, warnings, and failures. Do not force
a single winning classifier or count more classes as success by itself.

Ablate each family at the displayed method/K for both summary representations
and both encodings. Repeat all resample checks for each ablation. Compare
representations at matching method/K in a separate agreement audit.

Freeze input series, physical feature data, registry, source code, metrics,
transformers and candidate models before accessing external classification
labels. Köppen comparison is descriptive only. Bureau seasonal/humidity maps
inform interpretation; quantitative Bureau agreement requires a separately
verified local reference raster. No missing-reference metric is fabricated.

These rules were fixed before external evaluation of the new feature experiment.
The broader research catalogue includes variables requiring daily or new data;
those are not claimed to exist in the current 110-column registry.
