"""Named climate features, with physical values kept separate from model geometry."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import manifest, write_json
from .config import config_hash, project_path, run_path
from .preprocess import MONTHS, normalize_longitude


@dataclass
class FeatureTable:
    values: np.ndarray
    metadata: list[dict]

    def select(self, representation: str, omit: str | None = None) -> "FeatureTable":
        idx = [i for i, item in enumerate(self.metadata)
               if representation in item["representations"] and item["family"] != omit]
        if not idx:
            raise ValueError(f"No features for {representation}, omit={omit}")
        return FeatureTable(self.values[:, idx], [self.metadata[i] for i in idx])


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=np.abs(b) > 1e-10)


def circular_quarters(values: np.ndarray) -> np.ndarray:
    """Twelve three-month sums, indexed by their starting calendar month."""
    return sum(np.roll(values, -offset, axis=0) for offset in range(3))


def _correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return safe_ratio(np.mean(a * b, axis=0), np.sqrt(np.mean(a*a, axis=0)*np.mean(b*b, axis=0)))


def _windows(values: np.ndarray, years: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Exclude artificial joins introduced by temporal block resampling."""
    flat = values.reshape(-1, values.shape[-1])
    stamps = (years[:, None] * 12 + np.arange(12)).ravel()
    starts = np.arange(len(flat) - length + 1)
    valid = np.ones(len(starts), bool)
    for offset in range(1, length):
        valid &= stamps[starts + offset] == stamps[starts] + offset
    starts = starts[valid]
    sums = sum(flat[starts + offset] for offset in range(length))
    return sums, stamps[starts] % 12


def derive_features(temperature: np.ndarray, precipitation: np.ndarray, years: np.ndarray) -> FeatureTable:
    """Inputs have shape (year, calendar month, cell); temperature is monthly mean."""
    t, p, years = np.asarray(temperature, float), np.asarray(precipitation, float), np.asarray(years)
    if t.shape != p.shape or t.ndim != 3 or t.shape[:2] != (len(years), 12) or len(years) < 3:
        raise ValueError("Expected matching (at least 3 years, 12 months, cells) arrays")
    if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Source climate series must be complete, finite, and precipitation nonnegative")
    tc, pc = t.mean(axis=0), p.mean(axis=0)
    days = np.array([[calendar.monthrange(int(y), m)[1] for m in range(1, 13)] for y in years])
    ta = np.sum(t * days[:, :, None], axis=1) / days.sum(axis=1)[:, None]
    pa = p.sum(axis=1)
    annual_p = pc.sum(axis=0)
    frac = safe_ratio(pc, annual_p)
    cols, meta = [], []

    def add(name, value, family, units, definition, reps=("summaries", "variability"), transform="identity", source="project"):
        cols.append(value)
        meta.append(dict(name=name, family=family, units=units, definition=definition,
                         representations=list(reps), transform=transform, source=source))

    for m, month in enumerate(MONTHS):
        add(f"temp_{month}_c", tc[m], "temperature", "degC", "Mean monthly temperature over sampled years", ("baseline",))
        add(f"precip_{month}_mm", pc[m], "rainfall", "mm/month", "Mean monthly precipitation total", ("baseline",), "log1p")
        add(f"temp_anomaly_{month}", tc[m]-tc.mean(axis=0), "thermal_seasonality", "degC", "Monthly mean minus unweighted 12-month mean", ("profiles",))
        add(f"rain_fraction_{month}", frac[m], "rainfall_seasonality", "fraction", "Monthly share of annual climatological rain", ("profiles",))
    add("temperature_mean", ta.mean(axis=0), "temperature", "degC", "Mean of day-weighted annual mean temperatures", ("profiles", "summaries", "variability"))
    add("rainfall_mean", pa.mean(axis=0), "rainfall", "mm/year", "Mean annual precipitation", ("profiles", "summaries", "variability"), "log1p", "WorldClim BIO12")
    add("rainfall_median", np.median(pa, axis=0), "rainfall", "mm/year", "Median annual precipitation", transform="log1p", source="BOM seasonal rainfall")
    add("temperature_warmest_month_mean", tc.max(axis=0), "temperature", "degC", "Maximum monthly mean; NOT BIO5")
    add("temperature_coldest_month_mean", tc.min(axis=0), "temperature", "degC", "Minimum monthly mean; NOT BIO6")
    add("temperature_monthly_range", np.ptp(tc, axis=0), "thermal_seasonality", "degC", "Range of monthly means; NOT BIO7")
    add("temperature_seasonality", tc.std(axis=0), "thermal_seasonality", "degC", "Population SD of 12 monthly mean temperatures; BIO4 without x100", source="WorldClim BIO4")
    add("rainfall_wettest_month", pc.max(axis=0), "rainfall", "mm/month", "Maximum monthly climatological precipitation", transform="log1p", source="WorldClim BIO13")
    add("rainfall_driest_month", pc.min(axis=0), "rainfall", "mm/month", "Minimum monthly climatological precipitation", transform="log1p", source="WorldClim BIO14")
    add("rainfall_seasonality_cv", safe_ratio(pc.std(axis=0), pc.mean(axis=0)), "rainfall_seasonality", "ratio", "Population SD/mean of monthly precipitation; BIO15 without x100", source="WorldClim BIO15")
    add("rainfall_concentration", np.sum(frac**2, axis=0)*100, "rainfall_seasonality", "percent", "100 sum(P_month^2)/(sum P_month)^2")
    add("rainfall_entropy", -np.sum(np.where(frac > 0, frac*np.log(np.maximum(frac, 1e-30)), 0), axis=0)/np.log(12), "rainfall_seasonality", "fraction", "Normalized Shannon entropy of monthly fractions; all-zero rain undefined")
    cols[-1] = np.where(annual_p > 1e-10, cols[-1], np.nan)
    warm = [10, 11, 0, 1, 2, 3]
    add("warm_half_rainfall_fraction", np.sum(frac[warm], axis=0), "rainfall_seasonality", "fraction", "November-April share of mean annual precipitation", source="BOM seasonal rainfall (continuous summary)")
    add("summer_winter_rainfall_contrast", safe_ratio(pc[[11,0,1]].sum(axis=0)-pc[[5,6,7]].sum(axis=0), pc[[11,0,1,5,6,7]].sum(axis=0)), "rainfall_seasonality", "ratio", "(DJF-JJA)/(DJF+JJA)")
    for season, months in {"DJF":[11,0,1], "MAM":[2,3,4], "JJA":[5,6,7], "SON":[8,9,10]}.items():
        add(f"temperature_{season}", tc[months].mean(axis=0), "temperature", "degC", f"Unweighted climatological {season} monthly mean")
        add(f"rainfall_{season}", pc[months].sum(axis=0), "rainfall", "mm/season", f"Climatological {season} total", transform="log1p")
    tq, pq = circular_quarters(tc)/3, circular_quarters(pc)
    cells = np.arange(tc.shape[1])
    wet, dry, hot, cold = pq.argmax(axis=0), pq.argmin(axis=0), tq.argmax(axis=0), tq.argmin(axis=0)
    for name, arr, index, family, units, bio in [
        ("temperature_wettest_quarter", tq, wet, "coupling", "degC", 8),
        ("temperature_driest_quarter", tq, dry, "coupling", "degC", 9),
        ("temperature_warmest_quarter", tq, hot, "temperature", "degC", 10),
        ("temperature_coldest_quarter", tq, cold, "temperature", "degC", 11),
        ("rainfall_wettest_quarter", pq, wet, "rainfall", "mm/quarter", 16),
        ("rainfall_driest_quarter", pq, dry, "rainfall", "mm/quarter", 17),
        ("rainfall_warmest_quarter", pq, hot, "coupling", "mm/quarter", 18),
        ("rainfall_coldest_quarter", pq, cold, "coupling", "mm/quarter", 19),
    ]:
        add(name, arr[index, cells], family, units, "Circular three-month quarter; earliest starting month resolves ties", transform="log1p" if units.startswith("mm") else "identity", source=f"WorldClim BIO{bio}")
    for harmonic in (1, 2):
        angle = 2*np.pi*harmonic*np.arange(12)/12
        for name, basis in (("cos", np.cos(angle)), ("sin", np.sin(angle))):
            add(f"temperature_h{harmonic}_{name}", 2*np.mean((tc-tc.mean(axis=0))*basis[:, None], axis=0), "thermal_seasonality", "degC", f"Harmonic {harmonic} signed {name} coefficient; Jan phase zero")
            add(f"rainfall_h{harmonic}_{name}", np.sum(frac*basis[:, None], axis=0), "rainfall_seasonality", "fraction", f"Fraction-weighted harmonic {harmonic} {name}; magnitude encodes timing strength")
    add("temperature_rainfall_correlation", _correlation(tc, pc), "coupling", "correlation", "Pearson association of 12 climatological monthly means; undefined when either cycle constant")
    tx = np.sum((tc-tc.mean(axis=0))*np.exp(2j*np.pi*np.arange(12)/12)[:,None], axis=0)
    px = np.sum(frac*np.exp(2j*np.pi*np.arange(12)/12)[:,None], axis=0)
    phase = tx*np.conj(px)
    # Amplitude weighting avoids arbitrary phase jumps in near-uniform climates.
    add("seasonal_phase_alignment", safe_ratio(phase.real, np.sqrt(np.sum((tc-tc.mean(axis=0))**2,axis=0)*6)), "coupling", "ratio", "First harmonic alignment, weighted by rainfall concentration and thermal first-harmonic strength")
    for name, arr, family, units, definition in [
        ("annual_temperature_sd", ta.std(axis=0, ddof=1), "variability", "degC", "Sample SD of annual mean temperature"),
        ("annual_rainfall_sd", pa.std(axis=0, ddof=1), "variability", "mm/year", "Sample SD of annual rainfall"),
        ("annual_rainfall_cv", safe_ratio(pa.std(axis=0,ddof=1),pa.mean(axis=0)), "variability", "ratio", "Annual rainfall sample SD/mean"),
        ("annual_rainfall_mean_median_gap", safe_ratio(pa.mean(axis=0)-np.median(pa,axis=0),pa.mean(axis=0)), "variability", "ratio", "(Mean-median annual rainfall)/mean"),
    ]:
        add(name, arr, family, units, definition, ("variability",), "log1p" if units.startswith("mm") else "identity")
    for q in (.1,.5,.9):
        add(f"annual_rainfall_q{int(q*100)}_relative", safe_ratio(np.quantile(pa,q,axis=0),pa.mean(axis=0)), "variability", "ratio", "Annual rainfall quantile divided by annual mean", ("variability",))
    tw, start_month = _windows(t, years, 3)
    pw, _ = _windows(p, years, 3)
    for season, start in (("DJF",11),("MAM",2),("JJA",5),("SON",8)):
        mask = start_month == start
        if mask.sum() < 2:
            raise ValueError(f"Insufficient complete seasons for {season}")
        add(f"temperature_{season}_interannual_sd", (tw[mask]/3).std(axis=0,ddof=1), "variability", "degC", "Sample SD of complete seasonal monthly means", ("variability",))
        add(f"rainfall_{season}_interannual_cv", safe_ratio(pw[mask].std(axis=0,ddof=1),pw[mask].mean(axis=0)), "variability", "ratio", "Sample SD/mean of complete seasonal totals", ("variability",))
    for variable, arr in (("temperature",t),("rainfall",p)):
        anomalies = arr-arr.mean(axis=0)
        flat = anomalies.reshape(-1,arr.shape[-1])
        stamps = (years[:,None]*12+np.arange(12)).ravel()
        valid = np.diff(stamps)==1
        add(f"{variable}_anomaly_lag1", _correlation(flat[:-1][valid],flat[1:][valid]), "persistence", "correlation", "Lag-one monthly anomaly correlation; resampling seams excluded", ("variability",))
    for length in (3,6,12):
        totals, month = _windows(p,years,length)
        anomalies = totals.copy()
        for m in range(12):
            mask = month==m
            anomalies[mask] -= totals[mask].mean(axis=0)
        add(f"rainfall_{length}month_anomaly_relative_sd", safe_ratio(anomalies.std(axis=0,ddof=1),totals.mean(axis=0)), "persistence", "ratio", f"SD of calendar-deseasonalized {length}-month totals / mean total; seams excluded", ("variability",))
    # Rainfall below 50% of its own calendar-month normal: explicit proxy, not SPI or daily CDD.
    dry_month = p < .5*pc[None,:,:]
    run = np.zeros(p.shape[-1],int)
    maxrun = run.copy()
    counts = np.zeros_like(run)
    flat = dry_month.reshape(-1,p.shape[-1])
    stamps = (years[:,None]*12+np.arange(12)).ravel()
    for i, isdry in enumerate(flat):
        if i == 0 or stamps[i] != stamps[i-1]+1:
            run[:] = 0
        run = np.where(isdry,run+1,0)
        counts += run==3
        maxrun = np.maximum(maxrun,run)
    add("dry_month_max_run", maxrun, "persistence", "months", "Longest run below 50% of calendar-month mean rainfall; seams reset", ("variability",))
    add("dry_month_3plus_events_per_year", counts/len(years), "persistence", "events/year", "Runs reaching 3 dry months divided by sampled years; seam-censored proxy", ("variability",))
    return FeatureTable(np.column_stack(cols),meta)


class FeatureTransformer:
    """Fit imputation, correlation pruning, family balance and PCA only on training cells."""
    def __init__(self, use_pca: bool = False, correlation_threshold: float = .95, legacy: bool = False):
        self.use_pca, self.threshold, self.legacy = use_pca, correlation_threshold, legacy

    @staticmethod
    def _transform(table):
        x = table.values.copy()
        for i, meta in enumerate(table.metadata):
            if meta["transform"] == "log1p":
                if np.any(x[:,i] < 0):
                    raise ValueError(f"Negative value for {meta['name']}")
                x[:,i] = np.log1p(x[:,i])
            elif meta["transform"] != "identity":
                raise ValueError("Unknown feature transformation")
        return x

    def fit(self, table: FeatureTable, weights: np.ndarray):
        x = self._transform(table)
        w = np.asarray(weights,float)
        if len(w)!=len(x) or np.any(w<=0) or not np.isfinite(w).all():
            raise ValueError("Invalid training weights")
        w = w/w.sum()
        self.names = [m["name"] for m in table.metadata]
        finite = np.isfinite(x)
        self.mean = np.nan_to_num(safe_ratio(np.sum(np.where(finite,x,0)*w[:,None],axis=0),np.sum(finite*w[:,None],axis=0)))
        x = np.where(finite,x,self.mean)
        self.scale = np.sqrt(np.sum((x-self.mean)**2*w[:,None],axis=0))
        z = (x-self.mean)/np.maximum(self.scale,1e-12)
        keep, dropped = [], []
        for i, m in enumerate(table.metadata):
            reason = None
            if self.scale[i] < 1e-10:
                reason = "constant_or_all_missing"
            elif not self.legacy:
                for j in keep:
                    if table.metadata[j]["family"]==m["family"] and abs(np.sum(w*z[:,i]*z[:,j]))>self.threshold:
                        reason = f"correlated_with:{self.names[j]}"
                        break
            if reason:
                dropped.append(dict(name=m["name"],reason=reason))
            else:
                keep.append(i)
        if not keep:
            raise ValueError("No variable features remain")
        self.keep = np.array(keep)
        families = [table.metadata[i]["family"] for i in keep]
        self.balance = np.ones(len(keep)) if self.legacy else np.array([1/np.sqrt(families.count(f)) for f in families])
        balanced = z[:,self.keep]*self.balance
        self.components = None
        self.explained = 1.
        if self.use_pca:
            # Exact area-weighted covariance; no second stochastic area resample.
            eig, vec = np.linalg.eigh((balanced*w[:,None]).T@balanced)
            order = np.argsort(eig)[::-1]
            eig, vec = np.maximum(eig[order],0),vec[:,order]
            count = min(len(eig),np.searchsorted(np.cumsum(eig)/eig.sum(),.99)+1)
            self.components = vec[:,:count]
            self.explained = float(eig[:count].sum()/eig.sum())
        self.audit = dict(input_features=len(self.names), retained_features=len(keep),
                          output_dimensions=len(keep) if self.components is None else self.components.shape[1],
                          retained_names=[self.names[i] for i in keep], dropped=dropped,
                          missing_counts={self.names[i]:int((~finite[:,i]).sum()) for i in range(len(self.names)) if not finite[:,i].all()},
                          explained_variance=self.explained, family_balanced=not self.legacy)
        return self

    def transform(self, table: FeatureTable):
        if self.names != [m["name"] for m in table.metadata]:
            raise ValueError("Feature names/order differ from fitted schema")
        x = self._transform(table)
        x = np.where(np.isfinite(x),x,self.mean)
        z = ((x-self.mean)/np.maximum(self.scale,1e-12))[:,self.keep]*self.balance
        return z if self.components is None else z@self.components


def prepare_feature_series(config: dict, force: bool = False) -> Path:
    from .region import prepare_region
    regional = prepare_region(config,force)
    output = run_path(config,"data","processed","monthly_series.npz")
    if output.exists() and not force:
        return output
    with np.load(regional) as data:
        coords = {key:data[key] for key in ("latitude","longitude","cell_id","area_weights")}
        baseline = data["features"]
    years = np.arange(config["feature_expansion"]["years"][0],config["feature_expansion"]["years"][1]+1)
    temperatures, rainfalls, paths = [], [], []
    for year in years:
        path = project_path("data","interim",f"era5_land_{year}_0p5.npz")
        paths.append(path)
        with np.load(path) as data:
            lat,lon = data["latitude"],normalize_longitude(data["longitude"])
            ilat = np.argmin(abs(lat[:,None]-coords["latitude"]),axis=0)
            ilon = np.argmin(abs(lon[:,None]-coords["longitude"]),axis=0)
            if not np.allclose(lat[ilat],coords["latitude"],atol=1e-5,rtol=0) or not np.allclose(lon[ilon],coords["longitude"],atol=1e-5,rtol=0):
                raise ValueError(f"Regional cell alignment failed for {path}")
            temperatures.append(data["temperature_c"][:,ilat,ilon])
            rainfalls.append(data["precipitation_mm"][:,ilat,ilon])
    t,p = np.asarray(temperatures),np.asarray(rainfalls)
    if not np.allclose(np.concatenate([t.mean(axis=0).T,p.mean(axis=0).T],axis=1),baseline,rtol=2e-6,atol=1e-4):
        raise ValueError("Yearly data do not reproduce the regional baseline")
    table = derive_features(t,p,years)
    np.savez_compressed(output,temperature_c=t,precipitation_mm=p,years=years,**coords)
    physical = run_path(config,"data","processed","expanded_features.npz")
    np.savez_compressed(physical,features=table.values,feature_names=np.array([m["name"] for m in table.metadata]),**coords)
    metadata = run_path(config,"results","metrics","feature-catalogue.json")
    write_json(metadata,dict(features=table.metadata,count=len(table.metadata)))
    write_json(run_path(config,"results","manifests","feature-prepare.json"),manifest(
        "feature-prepare",config_hash(config),[regional,*paths,Path(__file__)],outputs=[str(output),str(physical)],cells=len(baseline)))
    return output
