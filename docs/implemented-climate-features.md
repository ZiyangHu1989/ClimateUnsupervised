# Implemented climate feature definitions

Generated from the named registry for the Australian feature-expansion experiment. Physical units are shown; log1p is applied only to model inputs. A column can belong to multiple representations. Undefined ratios/correlations are recorded as missing and imputed using training cells only.

## Temperature

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| temp_Jan_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Feb_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Mar_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Apr_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_May_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Jun_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Jul_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Aug_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Sep_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Oct_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Nov_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temp_Dec_c | degC | identity | baseline | Mean monthly temperature over sampled years; project |
| temperature_mean | degC | identity | profiles, summaries, variability | Mean of day-weighted annual mean temperatures; project |
| temperature_warmest_month_mean | degC | identity | summaries, variability | Maximum monthly mean; NOT BIO5; project |
| temperature_coldest_month_mean | degC | identity | summaries, variability | Minimum monthly mean; NOT BIO6; project |
| temperature_DJF | degC | identity | summaries, variability | Unweighted climatological DJF monthly mean; project |
| temperature_MAM | degC | identity | summaries, variability | Unweighted climatological MAM monthly mean; project |
| temperature_JJA | degC | identity | summaries, variability | Unweighted climatological JJA monthly mean; project |
| temperature_SON | degC | identity | summaries, variability | Unweighted climatological SON monthly mean; project |
| temperature_warmest_quarter | degC | identity | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO10 |
| temperature_coldest_quarter | degC | identity | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO11 |

## Rainfall

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| precip_Jan_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Feb_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Mar_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Apr_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_May_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Jun_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Jul_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Aug_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Sep_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Oct_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Nov_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| precip_Dec_mm | mm/month | log1p | baseline | Mean monthly precipitation total; project |
| rainfall_mean | mm/year | log1p | profiles, summaries, variability | Mean annual precipitation; WorldClim BIO12 |
| rainfall_median | mm/year | log1p | summaries, variability | Median annual precipitation; BOM seasonal rainfall |
| rainfall_wettest_month | mm/month | log1p | summaries, variability | Maximum monthly climatological precipitation; WorldClim BIO13 |
| rainfall_driest_month | mm/month | log1p | summaries, variability | Minimum monthly climatological precipitation; WorldClim BIO14 |
| rainfall_DJF | mm/season | log1p | summaries, variability | Climatological DJF total; project |
| rainfall_MAM | mm/season | log1p | summaries, variability | Climatological MAM total; project |
| rainfall_JJA | mm/season | log1p | summaries, variability | Climatological JJA total; project |
| rainfall_SON | mm/season | log1p | summaries, variability | Climatological SON total; project |
| rainfall_wettest_quarter | mm/quarter | log1p | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO16 |
| rainfall_driest_quarter | mm/quarter | log1p | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO17 |

## Thermal Seasonality

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| temp_anomaly_Jan | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Feb | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Mar | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Apr | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_May | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Jun | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Jul | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Aug | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Sep | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Oct | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Nov | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temp_anomaly_Dec | degC | identity | profiles | Monthly mean minus unweighted 12-month mean; project |
| temperature_monthly_range | degC | identity | summaries, variability | Range of monthly means; NOT BIO7; project |
| temperature_seasonality | degC | identity | summaries, variability | Population SD of 12 monthly mean temperatures; BIO4 without x100; WorldClim BIO4 |
| temperature_h1_cos | degC | identity | summaries, variability | Harmonic 1 signed cos coefficient; Jan phase zero; project |
| temperature_h1_sin | degC | identity | summaries, variability | Harmonic 1 signed sin coefficient; Jan phase zero; project |
| temperature_h2_cos | degC | identity | summaries, variability | Harmonic 2 signed cos coefficient; Jan phase zero; project |
| temperature_h2_sin | degC | identity | summaries, variability | Harmonic 2 signed sin coefficient; Jan phase zero; project |

## Rainfall Seasonality

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| rain_fraction_Jan | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Feb | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Mar | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Apr | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_May | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Jun | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Jul | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Aug | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Sep | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Oct | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Nov | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rain_fraction_Dec | fraction | identity | profiles | Monthly share of annual climatological rain; project |
| rainfall_seasonality_cv | ratio | identity | summaries, variability | Population SD/mean of monthly precipitation; BIO15 without x100; WorldClim BIO15 |
| rainfall_concentration | percent | identity | summaries, variability | 100 sum(P_month^2)/(sum P_month)^2; project |
| rainfall_entropy | fraction | identity | summaries, variability | Normalized Shannon entropy of monthly fractions; all-zero rain undefined; project |
| warm_half_rainfall_fraction | fraction | identity | summaries, variability | November-April share of mean annual precipitation; BOM seasonal rainfall (continuous summary) |
| summer_winter_rainfall_contrast | ratio | identity | summaries, variability | (DJF-JJA)/(DJF+JJA); project |
| rainfall_h1_cos | fraction | identity | summaries, variability | Fraction-weighted harmonic 1 cos; magnitude encodes timing strength; project |
| rainfall_h1_sin | fraction | identity | summaries, variability | Fraction-weighted harmonic 1 sin; magnitude encodes timing strength; project |
| rainfall_h2_cos | fraction | identity | summaries, variability | Fraction-weighted harmonic 2 cos; magnitude encodes timing strength; project |
| rainfall_h2_sin | fraction | identity | summaries, variability | Fraction-weighted harmonic 2 sin; magnitude encodes timing strength; project |

## Coupling

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| temperature_wettest_quarter | degC | identity | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO8 |
| temperature_driest_quarter | degC | identity | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO9 |
| rainfall_warmest_quarter | mm/quarter | log1p | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO18 |
| rainfall_coldest_quarter | mm/quarter | log1p | summaries, variability | Circular three-month quarter; earliest starting month resolves ties; WorldClim BIO19 |
| temperature_rainfall_correlation | correlation | identity | summaries, variability | Pearson association of 12 climatological monthly means; undefined when either cycle constant; project |
| seasonal_phase_alignment | ratio | identity | summaries, variability | First harmonic alignment, weighted by rainfall concentration and thermal first-harmonic strength; project |

## Variability

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| annual_temperature_sd | degC | identity | variability | Sample SD of annual mean temperature; project |
| annual_rainfall_sd | mm/year | log1p | variability | Sample SD of annual rainfall; project |
| annual_rainfall_cv | ratio | identity | variability | Annual rainfall sample SD/mean; project |
| annual_rainfall_mean_median_gap | ratio | identity | variability | (Mean-median annual rainfall)/mean; project |
| annual_rainfall_q10_relative | ratio | identity | variability | Annual rainfall quantile divided by annual mean; project |
| annual_rainfall_q50_relative | ratio | identity | variability | Annual rainfall quantile divided by annual mean; project |
| annual_rainfall_q90_relative | ratio | identity | variability | Annual rainfall quantile divided by annual mean; project |
| temperature_DJF_interannual_sd | degC | identity | variability | Sample SD of complete seasonal monthly means; project |
| rainfall_DJF_interannual_cv | ratio | identity | variability | Sample SD/mean of complete seasonal totals; project |
| temperature_MAM_interannual_sd | degC | identity | variability | Sample SD of complete seasonal monthly means; project |
| rainfall_MAM_interannual_cv | ratio | identity | variability | Sample SD/mean of complete seasonal totals; project |
| temperature_JJA_interannual_sd | degC | identity | variability | Sample SD of complete seasonal monthly means; project |
| rainfall_JJA_interannual_cv | ratio | identity | variability | Sample SD/mean of complete seasonal totals; project |
| temperature_SON_interannual_sd | degC | identity | variability | Sample SD of complete seasonal monthly means; project |
| rainfall_SON_interannual_cv | ratio | identity | variability | Sample SD/mean of complete seasonal totals; project |

## Persistence

| Name | Units | Model transform | Representations | Definition / source |
|---|---|---|---|---|
| temperature_anomaly_lag1 | correlation | identity | variability | Lag-one monthly anomaly correlation; resampling seams excluded; project |
| rainfall_anomaly_lag1 | correlation | identity | variability | Lag-one monthly anomaly correlation; resampling seams excluded; project |
| rainfall_3month_anomaly_relative_sd | ratio | identity | variability | SD of calendar-deseasonalized 3-month totals / mean total; seams excluded; project |
| rainfall_6month_anomaly_relative_sd | ratio | identity | variability | SD of calendar-deseasonalized 6-month totals / mean total; seams excluded; project |
| rainfall_12month_anomaly_relative_sd | ratio | identity | variability | SD of calendar-deseasonalized 12-month totals / mean total; seams excluded; project |
| dry_month_max_run | months | identity | variability | Longest run below 50% of calendar-month mean rainfall; seams reset; project |
| dry_month_3plus_events_per_year | events/year | identity | variability | Runs reaching 3 dry months divided by sampled years; seam-censored proxy; project |
