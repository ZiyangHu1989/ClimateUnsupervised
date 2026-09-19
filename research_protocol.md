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
