# Targeted data acquisition specification

This document specifies the next stage; the implemented feature experiment downloads no new climate data. First compare features derived from the existing temperature/rainfall series. Then add genuine moisture and daily-weather information in separate, traceable packages.

## Shared constraints

- **Period:** 1991-01-01 through 2020-12-31. Request the minimal 2021-01-01 accumulation endpoint when needed to complete the final daily rainfall total; exclude that endpoint from the normal period itself.
- **Coverage:** mainland Australia and Tasmania. Request a bounding box approximately north −10°, west 112°, south −45°, east 155°, then apply the existing pinned country mask. An input bounding box may contain surrounding ocean/countries; final cells must match the existing Australian cell IDs exactly.
- **Grid:** keep the existing 0.5° cell centres (~.2 fractional coordinates from the original native-grid coarsening). Do not assume a generic 0.5° CDS grid is aligned. Match the existing 5×5 native aggregation where available and verify coordinate equality. Otherwise explicitly remap cell areas onto the existing grid and record the method.
- **Units:** temperature °C after conversion; rainfall/ET mm per specified interval; radiation MJ m−2 per specified interval; wind m s−1; pressure kPa for FAO calculations. Record source units and conversions.
- **Data checks:** inspect one wet northern, one arid interior, one southwestern, and one Tasmanian location before a full request. Validate calendars, accumulated-variable conventions, UTC day boundaries, mask, missingness, and monthly totals against the current baseline.
- **Access:** CDS account/API credentials and acceptance of the relevant dataset terms are required. Never write secrets to manifests. Bureau product access, format and licensing must be verified for the requested product; published map visibility does not establish bulk-download availability.
- **Storage estimates below:** uncompressed float32 payloads, not provider transfer estimates. Compression, coordinates, metadata and temporary files are additional. No full global hourly download is required.

## Package A: humidity and atmospheric demand

### A1. Monthly humidity screening

Source: [ERA5-Land monthly means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview).

Request `2m_dewpoint_temperature` for each month of 1991–2020. Temperature already exists. Derive monthly approximate actual vapour pressure, RH and VPD; retain labels that identify calculations from monthly-mean inputs. Compute means, seasonal contrasts and variability; do not label their upper monthly quantiles daily extremes.

Expected retained payload: 360 × 2,770 × 4 bytes ≈ **4 MB per variable**. A 0.1° rectangular Australian source subset is approximately 0.15 million cells, or **0.22 GB per variable** over 360 months, before masking/coarsening. Stream one year at a time.

### A2. Reference evapotranspiration

Use [FAO-56 Penman–Monteith](https://www.fao.org/4/x0490e/x0490e06.htm) as the default named ET0 method. Acquire daily Tmin/Tmax, humidity, scalar wind speed, surface pressure, and radiation at compatible locations and times. Reuse Package B temperature when available.

Request ERA5-Land `2m_temperature`, `2m_dewpoint_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_pressure`, `surface_solar_radiation_downwards`, `surface_net_solar_radiation`, and `surface_net_thermal_radiation`, with hourly fields where needed to calculate daily scalar wind and humidity consistently. Use documented accumulated-variable handling for radiation. Assess whether model net radiation is suitable for the chosen reference-surface implementation; document albedo/reference-surface assumptions rather than silently mixing conventions.

Compute hourly scalar speed sqrt(u²+v²) before daily averaging; convert 10 m wind to 2 m using the FAO logarithmic profile assumption. Compute humidity from concurrent temperature/dew point before temporal averaging when possible. Preserve the FAO unit conversion tests and radiation sign convention. Calculate daily ET0 and aggregate to monthly/annual summaries. Monthly ET0, if used for an initial screening alternative, must be labelled as such and compared with daily-derived totals.

Do not use ERA5-Land `potential_evaporation` as interchangeable ET0. Its [documented convention](https://confluence.ecmwf.int/pages/viewpage.action?pageId=402639006) and [arid-condition limitations](https://codes.ecmwf.int/grib/param-db/228251) require a separate sensitivity result if acquired.

Expected retained daily inputs: 10,958 days × 2,770 × 4 bytes ≈ **0.121 GB per variable**; eight daily fields ≈ **0.97 GB**. Hourly native-grid source payloads are much larger: approximately **158 GB per variable** for the full rectangular 30-year subset. Process one month at a time, aggregate and validate, then discard reproducible hourly chunks. Budget roughly 0.45 GB per variable per month of temporary uncompressed native data, plus processing copies. Provider subsetting/regridding support must be tested before using a lower estimate.

**Go/no-go check:** monthly humidity has complete coverage and plausible coastal/interior contrasts; ET0 input conventions are verified and sample calculations match an independent FAO-56 implementation. If conventions fail, stop that package with a concrete diagnostic rather than substitute temperature-only PET without review.

## Package B: daily temperature and rainfall structure

### Temperature

The [ERA5-Land daily-statistics product](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview) can provide daily minimum, maximum and mean temperature. Use UTC days consistently for a national areal-climate analysis and document that Bureau observation-day definitions can differ.

Primary analysis: remap daily fields to the established 0.5° grid, then compute indices. This yields daily extrema of an areal field, not station extremes or true within-cell extrema. Sensitivity: calculate indices at native resolution and area-average those indices. Clearly distinguish the two products, especially threshold counts and spells.

Request each year separately, validate all 365/366 daily timestamps, and preserve provenance. Retained daily Tmin/Tmax/mean temperature is approximately **0.36 GB** for the masked grid. Native rectangular daily arrays are about **6.6 GB per variable** over the full record, before compression.

### Rainfall

**Important:** ERA5-Land daily statistics omit accumulated variables, including total precipitation. Do not request precipitation from that endpoint. Use [ERA5-Land hourly data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) and the official accumulation documentation to construct daily totals, or acquire a verified Bureau daily rainfall product as a separate observational experiment.

ERA5-Land forecast accumulations are not universally independent hourly increments. Implement explicit forecast-cycle/deaccumulation logic from the product documentation; verify the 00 UTC endpoint and month/year transitions with actual metadata before bulk acquisition. Monthly reconstructed totals must agree with the corresponding source monthly climatology within a documented numerical/aggregation tolerance, and any systematic discrepancy must be resolved before indices are calculated.

Derive wet-day frequency/intensity; Rx1day/Rx5day; CDD/CWD; heavy-rainfall counts; seasonal intermittency; and the northern September-to-following-August 50 mm onset. Only complete onset seasons lying inside the data period are used; incomplete boundary seasons are excluded and counted.

Retained masked daily rainfall is approximately **0.121 GB**. A full native hourly rectangular series could approach **158 GB** uncompressed, so stream and aggregate rather than retain the entire native record. Quantify final compressed storage from a representative month before starting the full series.

### Australian observational comparison

Inspect [Bureau gridded data availability](https://www.bom.gov.au/climate/averages/climatology/gridded-data-info/gridded_datasets_summary.shtml) for daily rainfall, Tmax/Tmin, vapour pressure and solar exposure. Treat these as an independently sourced sensitivity experiment: preserve dataset/version identity and remapping metadata. Do not fill gaps in ERA5 with a different product without explicitly modelling the resulting discontinuity.

## Package C: optional explanatory references

- Acquire matching-period [Bureau climate-classification grids](https://www.bom.gov.au/climate/maps/averages/climate-classification/) for descriptive comparison only. Preserve class labels/legends and nearest-neighbour categorical sampling; do not bilinearly interpolate class codes. Verify the exact class scheme and 30-year normal before calculating agreement.
- Consider [AWRA-L soil moisture/ET/runoff](https://awo.bom.gov.au/about/overview) as landscape-response validation. They are not independent climate-only measurements because land properties and model assumptions influence them.
- Archive ENSO/IOD/SAM indices and estimate local seasonal responses only as an exploratory, uncertainty-qualified layer. Identical global index values are not spatial clustering features.

## Acceptance and integration

Each package must deliver a source manifest, exact period/grid/cell IDs, named variables/units, missing-data audit, conversion tests, and physical-value arrays. Add one family at a time to the existing feature registry and repeat preprocessing-refitted cell/spatial/temporal checks and family ablations. Compare against the refreshed baseline; freeze the new experiment before accessing external class labels.

Do not mix periods, silently change resolution, or interpret finer-grained source data as evidence that 0.5° output resolves mountain or coastal microclimates. No success criterion prescribes a target number of clusters.
