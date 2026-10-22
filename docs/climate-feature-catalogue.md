# Climate features for unsupervised classification in Australia

This catalogue separates measurement of climate from the choice of clustering geometry. The objective is to distinguish general climate regimes in mainland Australia and Tasmania, using 1991–2020 and the existing 0.5° grid. A long feature list is a research inventory, not a recommendation to concatenate everything.

## Implemented inventory

See the [complete human-readable list of all 110 implemented columns](implemented-climate-features.md).

The first stage derives **110 distinct named columns**. Some columns appear in more than one representation: 24 baseline monthly climatologies; 26 separated level/seasonal-profile variables; 40 physical summaries; and 62 physical summaries plus variability/persistence variables. Exact per-column definitions, units, transformations, source attribution and representation membership are in [the machine-readable registry](../runs/australia-features/results/metrics/feature-catalogue.json). Original physical values are retained in `expanded_features.npz`.

| Family | Implemented measurements | Information / caution |
|---|---|---|
| Temperature | Day-weighted annual mean; hottest/coldest monthly means; DJF/MAM/JJA/SON means; hottest/coldest quarter means | Climate level; monthly means cannot measure daily temperature extremes |
| Thermal seasonality | Monthly anomalies; monthly range; monthly population SD; first/second harmonic cosine and sine coefficients | Annual and semiannual amplitude and timing; January has phase zero |
| Rainfall amount | Annual mean/median; monthly climatologies; seasonal totals; wettest/driest month and quarter | Physical rainfall magnitudes use log1p only in model inputs |
| Rainfall seasonality | Monthly fractions; monthly CV; precipitation concentration index; normalized Shannon entropy; November–April fraction; DJF–JJA contrast; first/second harmonic coefficients | Distinguishes seasonal shape from total rain; all-zero rainfall ratios undefined |
| Coupling | Temperature of wettest/driest quarter; rain in hottest/coldest quarter; monthly T–P correlation; amplitude-weighted harmonic alignment | Identifies coincidence of seasonal warmth and moisture |
| Variability | Annual T SD; annual rainfall SD/CV; relative annual 10th/50th/90th percentiles; normalized mean–median gap; seasonal T SD and rainfall CV | Uses the 30 individual years; 10th/90th percentiles describe this short sample, not return levels |
| Persistence | Lag-one monthly T/P anomaly correlation; calendar-deseasonalized rolling 3/6/12-month rainfall variability; longest below-half-normal monthly rainfall run; frequency of runs reaching three months | Monthly proxies; neither daily consecutive dry days nor SPI/SPEI; run lengths are censored at resampling seams |

Quarter calculations wrap December–January. If equally wet/dry/hot/cold quarters tie, the earliest starting calendar month wins. Temperature quarter and seasonal climatologies use unweighted monthly means; annual temperature uses calendar-day weights. Precipitation inputs already contain actual month lengths, including leap years, from the existing preprocessing stage.

Month-indexed phase angles are deliberately not used as Euclidean variables: December and January must be close. Signed harmonic components smoothly encode both strength and timing. The thermal–rainfall phase summary is amplitude weighted, so an almost absent seasonal signal does not acquire an arbitrary full-strength angle.

## Broader research catalogue

The entries below are candidate families for subsequent stages. Only the implemented inventory above is available in the current experiment. Counts should not be interpreted as independent dimensions.

### Bioclimatic temperature and rainfall (WorldClim / ANUCLIM lineage)

The [WorldClim definitions](https://www.worldclim.org/data/bioclim.html) provide nineteen established BIO variables. The existing monthly mean T/P data support annual and quarterly summaries and precipitation seasonality. Add monthly climatologies of daily Tmin and Tmax to calculate mean diurnal range (BIO2), isothermality (BIO3), maximum temperature of the warmest month (BIO5), minimum temperature of the coldest month (BIO6), and the corresponding annual range (BIO7).

Do not label the range of monthly **mean** temperatures BIO7. BIO4 in this project is stored in °C SD, not the conventional SD ×100. BIO15 is a ratio, not CV ×100. Those constant multipliers do not change standardized distance, but units must remain explicit. WorldClim supplies definitions here; its gridded climatology is not silently mixed into the 1991–2020 ERA5-Land data.

### Australian seasonal rainfall regimes

Candidates: median warm-half and cool-half rain; their ratio; summer/winter incidence; relative wettest-quarter contribution; six-month rain concentration; dry-season length; wet-season length; seasonal asymmetry; annual/semiannual harmonic amplitude ratio; weighted circular concentration; interannual variability of onset and cessation; onset failures; rainfall accumulated after onset.

The [Bureau of Meteorology classification](https://www.bom.gov.au/climate/maps/averages/climate-classification/) uses median rainfall and November–April versus May–October seasonal incidence. Our November–April **mean rainfall fraction** is a continuous summary inspired by that distinction, not a reproduction of Bureau classes. Exact reproduction requires the published median-based method and matching reference data. A national dry/wet-season threshold is not assumed universally suitable across northern, southern and arid Australia.

The Bureau's [northern rainfall onset](https://www.bom.gov.au/climate/rainfall-onset/archive/20260730.archive.shtml) is the first accumulation of 50 mm after 1 September. Implement it as a northern-regime descriptor once daily precipitation exists; retain no-onset years separately, rather than encode them as an arbitrary date.

### Humidity and atmospheric moisture demand

Candidates: mean and seasonal dew point; actual vapour pressure; relative humidity; specific humidity; vapour-pressure deficit (VPD); daytime VPD; upper VPD quantiles; months/days with high VPD; humidity seasonal amplitude; warm-season humidity; co-occurrence of high temperature and low humidity; wet-bulb temperature and humid-heat frequency.

[ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview) provides dew point and temperature. Derived RH/VPD from monthly means are approximations: nonlinear saturation-vapour-pressure equations mean that a function of monthly means is not a monthly mean of the hourly function. Wet-bulb extremes and compound heat–humidity events require concurrent finer-time data. The Bureau separately classifies temperature and humidity, demonstrating why rainfall alone is not a complete moisture descriptor.

### Evaporation demand and climatic water balance

Candidates: annual and seasonal FAO-56 reference evapotranspiration (ET0); precipitation/ET0 aridity ratio; annual/seasonal P−ET0; sum of positive monthly deficits max(ET0−P,0); sum of surpluses; number and longest sequence of deficit months; deficit timing; interannual variability of deficit; atmospheric demand seasonality; warm-season water stress.

Use [FAO-56](https://www.fao.org/4/x0490e/x0490e06.htm) for a named reference-surface method and document its input/time-step requirements. Radiation, humidity and wind add information beyond temperature. Do not treat temperature-only PET estimates as an independent observational feature or use them as the default moisture-demand package.

ECMWF's [ERA5-Land documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=402639006) distinguishes its potential-evaporation convention from reference crop ET0. The [parameter documentation](https://codes.ecmwf.int/grib/param-db/228251) also warns about unrealistic potential evaporation under arid conditions. That matters directly for Australia; the model variable is a sensitivity comparison, not our default ET0.

### Daily thermal regime and extremes

Candidates following [Climdex](https://www.climdex.org/learn/indices/): daily temperature range (DTR); annual/monthly highest daily Tmax (TXx), lowest Tmax (TXn), highest Tmin (TNx), lowest Tmin (TNn); frost days (FD); summer days (SU); tropical nights (TR); cold/warm day and night fractions (TX10p/TX90p/TN10p/TN90p); warm/cold spell duration (WSDI/CSDI). Also catalogue days above 35°C and 40°C, seasonal heatwave frequency/duration/intensity, frost-season timing, and annual variability of these counts.

Use published thresholds and spell definitions for standard names. User-defined thresholds need distinct names. Local percentile exceedance rates in their own calibration period are partly standardized by construction; absolute thresholds, intensities and persistence may be more informative for spatial classification. Growing-degree days, chilling hours, growing-season length, heating/cooling degree days and thermal comfort indices are secondary ecological/building-use features, not default general-regime inputs.

### Daily rain occurrence, intensity and extremes

Candidates: wet days ≥1 mm; simple daily intensity index (SDII); heavy-rain days ≥10/20 mm; user-defined ≥50 mm counts; maximum one-day/five-day rain (Rx1day/Rx5day); very/extremely wet-day totals (R95p/R99p); fraction of annual rain from heavy events; consecutive dry/wet days (CDD/CWD); wet/dry transition probabilities; wet-spell intensity; event concentration; within-season rainfall intermittency; seasonal variability of all these quantities.

The [Climdex project](https://www.climdex.org/learn/) defines these from daily observations or modelled data. Preserve its exact wet-day, percentile-baseline and spell conventions. Spatially averaging daily precipitation before thresholding defines an areal climate index; averaging indices calculated at native cells answers a different question. Both can be useful, but must not be conflated.

### Drought and compound events

Catalogue SPI and SPEI at 1/3/6/12/24 months, drought duration/severity/recovery, soil-moisture drought, joint heat–dryness days, hot–humid days, and concurrent low-rain/high-demand seasons. A new implementation must specify distribution fitting, calibration baseline, zero treatment and uncertainty before these become classifier inputs.

Do not cluster on long-term mean standardized drought scores: standardization removes much of the desired between-location contrast. Similarly, in-baseline percentile exceedance frequencies can be approximately fixed by construction. Event persistence, absolute deficit and cross-variable dependence can be more informative. Thirty years is inadequate to claim well-constrained rare-event return periods everywhere.

### Radiation, wind and circulation

Candidates: downward shortwave radiation; net surface radiation; seasonal radiation amplitude; radiation relative to extraterrestrial radiation (clearness proxy); wind-speed means/quantiles; seasonal wind variability; directional persistence; vector seasonal changes; surface-pressure variability; synoptic regime occurrence; cloud cover; convective versus large-scale rainfall fractions; cyclone/front contribution to rain.

Some variables require ERA5 atmospheric fields rather than ERA5-Land. Mean u/v wind components cannot recover mean scalar wind speed. Cloud, convective diagnostics, fronts and cyclone attribution require separate source/method validation and remain exploratory. Elevation and distance to coast explain mechanisms but are not climate observations and stay out of the primary classifier.

### Land-surface and ecological responses

Candidates: surface/root-zone soil moisture; seasonal soil-moisture amplitude; low-moisture duration; actual ET; ET0−actual ET; evaporative fraction; runoff coefficient; soil-water recharge; snow-cover fraction/duration; snow-water equivalent; vegetation greenness/phenology and productivity.

[AWRA-L](https://awo.bom.gov.au/about/overview) models landscape water stores and fluxes. These depend on soils, vegetation and modelling assumptions in addition to meteorological climate. Keep them as optional validation or a separately labelled hydroclimate classification. Avoid vegetation-class inputs when vegetation-based climate maps are later used as external references.

### Interannual climate-driver response and change

Candidates: seasonal local temperature/rainfall response to ENSO, IOD and SAM; lagged responses; conditional seasonal means; explained anomaly variance; response consistency across record halves. The [Bureau describes the seasonally varying role of SAM and other drivers](https://www.bom.gov.au/resources/learn-and-explore/climate-knowledge-centre/climate-factors/southern-annular-mode).

Global index time series themselves are identical at every cell and therefore cannot distinguish locations; local response coefficients can. Responses must account for autocorrelation, covariance between drivers and a small effective sample size. Climate trends, trend acceleration, change in extremes and classification migration are a separate change-analysis layer; adding them to the primary normal-climate classifier would change its scientific question.

## Selection and validation policy

1. Keep physical values and transformations separate. Each named column has one family and explicit units.
2. In engineered representations, impute undefined values using training-area means; drop constant/all-missing features; prune absolute weighted within-family correlations >0.95 in registry order. Save every decision. Baseline retains the original monthly variables without pruning.
3. Scale to weighted unit variance, then divide each retained engineered family by the square root of its column count. Thus every family contributes equal total training variance. This is an explicit scientific prior, not an automatic discovery of feature importance.
4. Compare balanced direct geometry with 99%-variance area-weighted PCA. Perform all fitting and pruning inside each spatial/cell resample.
5. Recompute features in temporal block resamples. Preserve continuity only inside genuine chronological segments.
6. Compare representations and family ablations; retain the existing silhouette/area rules as auditable benchmarks, including smaller-area sensitivity. Do not equate more clusters with improved climate science.
7. Freeze candidate results before descriptive external evaluation. Köppen/Bureau labels never choose variables or K.

External sources were checked on 2026-09-20. Dataset catalogue contents and access terms may change; recheck at the subsequent acquisition stage.
