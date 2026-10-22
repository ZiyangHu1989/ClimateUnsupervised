"""Isolated, restartable experiment with preprocessing refit in every stability trial."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score

from .artifacts import manifest, verify_manifest_inputs, write_json
from .config import config_hash, project_path, run_path
from .feature_engineering import FeatureTable, FeatureTransformer, derive_features, prepare_feature_series
from .modeling import area_resample, normalize_labels, spatial_block_folds
from .multiscale import METHODS, FAMILIES, _fit_predict, _pairwise_summary, build_shortlist


def resample_years(years: np.ndarray, seed: int, block: int) -> np.ndarray:
    """Moving blocks without wrapping the end of the actual record."""
    if block < 1 or block > len(years):
        raise ValueError("Invalid temporal block length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,len(years)-block+1,size=int(np.ceil(len(years)/block)))
    return np.concatenate([np.arange(s,s+block) for s in starts])[:len(years)]


def _subset(table, idx):
    return FeatureTable(table.values[idx],table.metadata)


def make_trials(table, temporal_tables, weights, latitude, longitude, config, pca, legacy):
    s, f = config["multiscale"],config["feature_expansion"]
    def fit(source, idx, w):
        transform = FeatureTransformer(pca,float(f["correlation_threshold"]),legacy).fit(_subset(source,idx),w)
        return transform,transform.transform(source)
    base,values = fit(table,np.arange(len(weights)),weights)
    trials = {"bootstrap":[],"spatial":[],"temporal":[]}
    for seed in s["seeds"]:
        pool = area_resample(weights,min(s["algorithm_sample_max"],len(weights)),10000+seed)
        tr,x = fit(table,pool,np.ones(len(pool)))
        trials["bootstrap"].append((x,pool,True,seed,tr.audit))
    folds = spatial_block_folds(latitude,longitude,s["spatial_block_degrees"],s["spatial_folds"])
    for fold in range(s["spatial_folds"]):
        pool = np.flatnonzero(folds!=fold)
        tr,x = fit(table,pool,weights[pool])
        trials["spatial"].append((x,pool,False,20000+fold,tr.audit))
    for seed,source in zip(f["temporal_seeds"],temporal_tables):
        tr,x = fit(source,np.arange(len(weights)),weights)
        trials["temporal"].append((x,None,False,seed,tr.audit))
    return base,values,trials


def _agreement_to_reference(reference, labels, weights):
    """Per-cell agreement after optimal area-weighted permutation of cluster IDs."""
    unique_ref,unique_run = np.unique(reference),np.unique(labels)
    count = np.zeros((len(unique_ref),len(unique_run)))
    for i,a in enumerate(unique_ref):
        for j,b in enumerate(unique_run):
            count[i,j] = weights[(reference==a)&(labels==b)].sum()
    a,b = linear_sum_assignment(-count)
    mapping = {unique_run[j]:unique_ref[i] for i,j in zip(a,b)}
    return np.array([mapping.get(x,-1) for x in labels])==reference


def fit_candidate(method,k,values,trials,weights,config):
    s,f = config["multiscale"],config["feature_expansion"]
    seed = config["project"]["random_seed"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model,raw = _fit_predict(method,k,values,weights,seed,sample_max=s["algorithm_sample_max"],n_init=s["final_n_init"])
        labels,mapping = normalize_labels(raw,weights)
        if len(np.unique(labels)) != k:
            raise ValueError(f"Requested {k} clusters, obtained {len(np.unique(labels))}")
        stability,agreements = {},[]
        for kind,runs in trials.items():
            results=[]
            for x,pool,resampled,run_seed,_ in runs:
                _,pred = _fit_predict(method,k,x,weights,run_seed,pool=pool,
                                      sample_max=s["algorithm_sample_max"],n_init=f["stability_n_init"],pool_is_resample=resampled)
                if len(np.unique(pred)) != k:
                    raise ValueError(f"{kind} trial collapsed to {len(np.unique(pred))} clusters")
                results.append(pred)
                agreements.append(_agreement_to_reference(labels,pred,weights))
            stability[kind+"_stability"] = _pairwise_summary(results)
        shares = np.array([weights[labels==label].sum()/weights.sum() for label in np.unique(labels)])
        row = dict(method=method,family=FAMILIES[method],k=k,
                   silhouette=float(silhouette_score(values,labels,sample_size=min(len(values),f["silhouette_sample_max"]),random_state=seed)),
                   davies_bouldin=float(davies_bouldin_score(values,labels)),
                   minimum_cluster_area_fraction=float(shares.min()),area_shares=shares.tolist(),
                   label_mapping=mapping, warnings=sorted({str(w.message) for w in caught}),**stability)
    row["eligible"] = bool(row["bootstrap_stability"]["mean"]>=s["minimum_stability_ari"] and shares.min()>=s["minimum_cluster_area_fraction"])
    return row,labels,model,np.mean(agreements,axis=0).astype(np.float32)


def shortlist(rows,labels,config,minimum_area=None):
    s = config["multiscale"]
    minimum_area = s["minimum_cluster_area_fraction"] if minimum_area is None else minimum_area
    candidates = [dict(row,eligible=row["bootstrap_stability"]["mean"]>=s["minimum_stability_ari"] and row["minimum_cluster_area_fraction"]>=minimum_area) for row in rows if "error" not in row]
    if not candidates:
        return []
    agreement={}
    for k in sorted({row["k"] for row in candidates}):
        eligible=[row for row in candidates if row["k"]==k and row["eligible"]]
        scores=[]
        for i,left in enumerate(eligible):
            for right in eligible[i+1:]:
                if left["family"]!=right["family"]:
                    scores.append(adjusted_rand_score(labels[f"{left['method']}_k{k}"],labels[f"{right['method']}_k{k}"]))
        if scores:
            agreement[k]=float(np.median(scores))
    selected=build_shortlist(candidates,agreement,s["silhouette_retention_ratio"],s["minimum_stability_ari"],s["minimum_cross_algorithm_ari"],s["strong_single_family_stability_ari"],s["minimum_spatial_stability_ari"])
    for item in selected:
        row=next(r for r in candidates if r["k"]==item["k"] and r["method"]==item["representative_method"])
        item["temporal_confirmed"]=row["temporal_stability"]["mean"]>=config["feature_expansion"]["minimum_temporal_stability_ari"]
    return selected


def representative(rows,selected):
    valid=[r for r in rows if "error" not in r]
    confirmed={(r["representative_method"],r["k"]) for r in selected if r["temporal_confirmed"]}
    eligible=[r for r in valid if (r["method"],r["k"]) in confirmed]
    if eligible:
        # Choose the most detailed passing scale for display; retain all candidates.
        return max(eligible,key=lambda r:(r["k"],r["bootstrap_stability"]["mean"])),"temporally_confirmed"
    return max(valid,key=lambda r:min(r[x+"_stability"]["mean"] for x in ("bootstrap","spatial","temporal"))),"diagnostic_only"


def run_feature_experiment(config: dict, force: bool = False) -> Path:
    series=prepare_feature_series(config,force)
    with np.load(series) as data:
        t,p,years=data["temperature_c"],data["precipitation_mm"],data["years"]
        w,lat,lon=data["area_weights"].astype(float),data["latitude"],data["longitude"]
    f,s=config["feature_expansion"],config["multiscale"]
    full=derive_features(t,p,years)
    temporal=[]
    for seed in f["temporal_seeds"]:
        idx=resample_years(years,seed,f["temporal_block_years"])
        temporal.append(derive_features(t[idx],p[idx],years[idx]))
    metrics_dir=run_path(config,"results","metrics")
    models_dir=run_path(config,"results","models")
    models_dir.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),Path(__file__).with_name("feature_engineering.py"),Path(__file__).with_name("multiscale.py"),Path(__file__).with_name("modeling.py")]
    provenance=manifest("feature-experiment",config_hash(config),[series,*source_files])
    fingerprint=dict(config_sha256=provenance["config_sha256"],inputs=provenance["inputs"])
    all_runs={}
    frozen_inputs=[series,*source_files,run_path(config,"data","processed","expanded_features.npz"),metrics_dir/"feature-catalogue.json",run_path(config,"results","manifests","feature-prepare.json")]
    for rep in f["representations"]:
        table=full.select(rep)
        temps=[x.select(rep) for x in temporal]
        for encoding in f["encodings"]:
            name=f"{rep}-{encoding}"
            print(f"feature_experiment representation={name}",flush=True)
            base,values,trials=make_trials(table,temps,w,lat,lon,config,encoding=="pca",rep=="baseline")
            joblib.dump(base,models_dir/f"{name}-transform.joblib")
            rows,labels=[],{}
            for k in range(s["k_min"],s["k_max"]+1):
                for method in METHODS:
                    key=f"{method}_k{k}"
                    checkpoint=metrics_dir/f"{name}-{key}.json"
                    artifact=models_dir/f"{name}-{key}.joblib"
                    cache=json.loads(checkpoint.read_text()) if checkpoint.exists() and not force else None
                    if cache is not None and cache.get("fingerprint")!=fingerprint:
                        raise RuntimeError("Stale experiment checkpoint; use a new output root or --force")
                    if cache is not None and ("error" in cache["row"] or artifact.exists()):
                        row=cache["row"]
                        if "error" not in row:
                            labels[key]=joblib.load(artifact)["labels"]
                    else:
                        try:
                            row,pred,model,confidence=fit_candidate(method,k,values,trials,w,config)
                            labels[key]=pred
                            joblib.dump(dict(model=model,labels=pred,agreement=confidence),artifact)
                        except ValueError as exc:
                            row=dict(method=method,family=FAMILIES[method],k=k,error=str(exc))
                        write_json(checkpoint,dict(fingerprint=fingerprint,row=row))
                    rows.append(row)
                print(f"feature_experiment representation={name} completed_k={k}",flush=True)
            selected=shortlist(rows,labels,config)
            chosen,status=representative(rows,selected)
            result=dict(representation=rep,encoding=encoding,transform=base.audit,
                        resample_transforms={kind:[r[-1] for r in runs] for kind,runs in trials.items()},
                        candidates=rows,shortlist=selected,
                        area_sensitivity={str(a):shortlist(rows,labels,config,a) for a in f["area_sensitivity"]},
                        display=dict(method=chosen["method"],k=chosen["k"],status=status))
            output=metrics_dir/f"{name}.json"
            write_json(output,result)
            frozen_inputs.extend([output,models_dir/f"{name}-transform.joblib"])
            frozen_inputs.extend(models_dir/f"{name}-{key}.joblib" for key in labels)
            all_runs[name]=result

    # Ablate each family at each summary representation's displayed resolution, with the same trials.
    ablations=[]
    for name,result in all_runs.items():
        if result["representation"] not in ("summaries","variability"):
            continue
        rep,encoding=result["representation"],result["encoding"]
        display=result["display"]
        method,k=display["method"],display["k"]
        reference=joblib.load(models_dir/f"{name}-{method}_k{k}.joblib")["labels"]
        for family in sorted({m["family"] for m in full.select(rep).metadata}):
            key=f"{name}-without-{family}"
            path=metrics_dir/f"ablation-{key}.json"
            cached=json.loads(path.read_text()) if path.exists() and not force else None
            if cached is not None:
                if cached.get("fingerprint")!=fingerprint:
                    raise RuntimeError("Stale ablation checkpoint")
                row=cached["row"]
            else:
                base,values,trials=make_trials(full.select(rep,family),[x.select(rep,family) for x in temporal],w,lat,lon,config,encoding=="pca",False)
                try:
                    row,pred,_,_=fit_candidate(method,k,values,trials,w,config)
                    row.update(ari_to_full=float(adjusted_rand_score(reference,pred)),retained_features=base.audit["retained_features"])
                except ValueError as exc:
                    row=dict(error=str(exc),method=method,k=k)
                row.update(representation=name,omitted_family=family,reference_status=display["status"])
                write_json(path,dict(fingerprint=fingerprint,row=row))
            ablations.append(row)
            frozen_inputs.append(path)
            print(f"feature_ablation complete={key}",flush=True)
    # Cross-representation agreement uses the same method/K to avoid conflating resolution.
    comparisons=[]
    names=list(all_runs)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            for method in METHODS:
                for k in range(s["k_min"],s["k_max"]+1):
                    a,b=[models_dir/f"{n}-{method}_k{k}.joblib" for n in (left,right)]
                    if a.exists() and b.exists():
                        comparisons.append(dict(left=left,right=right,method=method,k=k,ari=float(adjusted_rand_score(joblib.load(a)["labels"],joblib.load(b)["labels"]))))
    summary=metrics_dir/"feature-experiment.json"
    write_json(summary,dict(fingerprint=fingerprint,runs=all_runs,ablations=ablations,cross_representation=comparisons,
                            external_labels_used=False,temporal_resampling="5-year moving blocks, artificial seams excluded"))
    frozen_inputs.append(summary)
    frozen=run_path(config,"results","manifests","feature-frozen.json")
    write_json(frozen,manifest("feature-freeze",config_hash(config),frozen_inputs,external_labels_used=False))
    return summary


def evaluate_feature_experiment(config: dict) -> Path:
    from .feature_reporting import report_feature_experiment
    frozen_path=run_path(config,"results","manifests","feature-frozen.json")
    frozen=json.loads(frozen_path.read_text())
    if frozen["config_sha256"]!=config_hash(config):
        raise RuntimeError("Configuration differs from frozen experiment")
    verify_manifest_inputs(frozen)
    output=report_feature_experiment(config)
    verify_manifest_inputs(frozen)
    write_json(run_path(config,"results","manifests","feature-evaluate.json"),manifest("feature-evaluate",config_hash(config),[frozen_path,project_path("data","external","koppen_geiger_1991_2020.tif")],report=str(output)))
    return output
