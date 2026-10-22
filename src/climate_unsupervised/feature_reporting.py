"""Scientific figures and descriptive external evaluation, only after selection freeze."""
from __future__ import annotations

import csv
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from .artifacts import write_json
from .config import project_path, run_path
from .evaluate import _scores, sample_raster, koppen_primary_groups
from .multiscale import METHODS, _save_figure


def report_feature_experiment(config: dict):
    root=run_path(config,"results")
    metrics=json.loads((root/"metrics"/"feature-experiment.json").read_text())
    catalogue=json.loads((root/"metrics"/"feature-catalogue.json").read_text())["features"]
    with np.load(run_path(config,"data","processed","expanded_features.npz")) as data:
        features=data["features"]
        lat,lon,w=data["latitude"],data["longitude"],data["area_weights"]
        names=list(data["feature_names"])
    with np.load(run_path(config,"data","processed","monthly_series.npz")) as data:
        t,p=data["temperature_c"].mean(axis=0),data["precipitation_mm"].mean(axis=0)
    ref,valid=sample_raster(project_path("data","external","koppen_geiger_1991_2020.tif"),lon,lat)
    primary=koppen_primary_groups(ref)
    valid &= primary>=0
    figures=root/"figures"/"features"
    figures.mkdir(parents=True,exist_ok=True)
    external,profiles={},[]
    comparison_rows=[]
    fig,axes=plt.subplots(2,4,figsize=(19,10),sharex=True,sharey=True)
    ufig,uaxes=plt.subplots(2,4,figsize=(19,10),sharex=True,sharey=True)
    for ax,uax,(name,result) in zip(axes.flat,uaxes.flat,metrics["runs"].items()):
        d=result["display"]
        k,method=d["k"],d["method"]
        artifact=joblib.load(root/"models"/f"{name}-{method}_k{k}.joblib")
        labels,agreement=artifact["labels"],artifact["agreement"]
        cmap=plt.get_cmap("tab20",k)
        norm=BoundaryNorm(np.arange(k+1)-.5,k)
        sc=ax.scatter(lon,lat,c=labels,s=7,marker="s",cmap=cmap,norm=norm,linewidths=0,rasterized=True)
        fig.colorbar(sc,ax=ax,ticks=np.arange(k),shrink=.75)
        ax.set_title(f"{name}: K={k}, {method}\n{d['status'].replace('_',' ')}",fontsize=10)
        im=uax.scatter(lon,lat,c=agreement,s=7,marker="s",cmap="viridis",vmin=0,vmax=1,linewidths=0,rasterized=True)
        uax.set_title(name,fontsize=10)
        for a in (ax,uax):
            a.set_aspect(1/np.cos(np.deg2rad(-27)))
            a.set(xlabel="Longitude",ylabel="Latitude")
        # All selected resolutions, plus explicitly tagged diagnostic displays.
        targets={(x["representative_method"],x["k"]) for x in result["shortlist"]}
        targets.add((method,k))
        external[name]={}
        for m,n in sorted(targets):
            lab=joblib.load(root/"models"/f"{name}-{m}_k{n}.joblib")["labels"]
            external[name][f"{m}_k{n}"]=dict(primary_groups=_scores(lab[valid],primary[valid],w[valid]),subclasses=_scores(lab[valid],ref[valid],w[valid]))
        row=next(x for x in result["candidates"] if x["method"]==method and x["k"]==k)
        ex=external[name][f"{method}_k{k}"]
        comparison_rows.append(dict(name=name,method=method,k=k,status=d["status"],features=result["transform"]["retained_features"],dimensions=result["transform"]["output_dimensions"],bootstrap=row["bootstrap_stability"]["mean"],spatial=row["spatial_stability"]["mean"],temporal=row["temporal_stability"]["mean"],minimum_area=row["minimum_cluster_area_fraction"],subclass_ari=ex["subclasses"]["unweighted"]["ari"]))
        pf,pax=plt.subplots(1,3,figsize=(16,4.5))
        for label in np.unique(labels):
            mask=labels==label
            area=float(w[mask].sum()/w.sum())
            color=cmap(int(label))
            pax[0].plot(range(1,13),np.average(t[:,mask],axis=1,weights=w[mask]),color=color,label=f"{label}: {area:.1%}")
            pax[1].plot(range(1,13),np.average(p[:,mask],axis=1,weights=w[mask]),color=color)
            for j,fname in enumerate(("annual_rainfall_cv","rainfall_DJF_interannual_cv","rainfall_JJA_interannual_cv")):
                values=features[mask,names.index(fname)]
                ok=np.isfinite(values)
                pax[2].bar(j+(int(label)-(k-1)/2)*.8/k,np.average(values[ok],weights=w[mask][ok]),width=.8/k,color=color)
            for j,meta in enumerate(catalogue):
                values=features[mask,j]
                ok=np.isfinite(values)
                profiles.append(dict(representation=name,method=method,k=k,cluster=int(label),area_fraction=area,feature=meta["name"],units=meta["units"],mean=float(np.average(values[ok],weights=w[mask][ok])) if ok.any() else None))
        pax[0].set(title="Monthly mean temperature",xlabel="Month",ylabel="°C",xticks=range(1,13))
        pax[1].set(title="Monthly precipitation",xlabel="Month",ylabel="mm/month",xticks=range(1,13))
        pax[2].set(title="Rainfall variability",ylabel="Coefficient of variation",xticks=range(3),xticklabels=["Annual","DJF","JJA"])
        pax[0].legend(fontsize=8,ncol=2)
        pf.suptitle(f"{name}: K={k}, {method} ({d['status']})")
        _save_figure(pf,figures/f"profiles-{name}.png")
    fig.suptitle("Australia: representative classifications (cluster colours are local to each panel)")
    _save_figure(fig,figures/"classification-comparison.png")
    ufig.subplots_adjust(right=.9)
    cax=ufig.add_axes([.93,.2,.015,.6])
    ufig.colorbar(im,cax=cax,label="Permutation-matched resample agreement")
    ufig.suptitle("Assignment reproducibility across cell, spatial, and temporal resamples; not a probability")
    ufig.savefig(figures/"assignment-agreement.png",dpi=180,bbox_inches="tight")
    plt.close(ufig)

    fig,axes=plt.subplots(2,4,figsize=(19,8),sharex=True,sharey=True)
    selected_maps=["temperature_mean","temperature_monthly_range","rainfall_mean","warm_half_rainfall_fraction","rainfall_entropy","annual_rainfall_cv","rainfall_JJA_interannual_cv","rainfall_anomaly_lag1"]
    for ax,name in zip(axes.flat,selected_maps):
        values=features[:,names.index(name)]
        im=ax.scatter(lon,lat,c=values,s=7,marker="s",cmap="viridis",linewidths=0,rasterized=True)
        meta=catalogue[names.index(name)]
        fig.colorbar(im,ax=ax,shrink=.8,label=meta["units"])
        ax.set(title=name.replace("_"," "),xlabel="Longitude",ylabel="Latitude")
        ax.set_aspect(1/np.cos(np.deg2rad(-27)))
    _save_figure(fig,figures/"physical-feature-maps.png")

    fig,axes=plt.subplots(len(metrics["runs"]),3,figsize=(15,3*len(metrics["runs"])),sharex=True,sharey=True)
    for axs,(name,result) in zip(axes,metrics["runs"].items()):
        for ax,kind in zip(axs,("bootstrap","spatial","temporal")):
            for method in METHODS:
                rows=[x for x in result["candidates"] if x["method"]==method and "error" not in x]
                ax.plot([x["k"] for x in rows],[x[kind+"_stability"]["mean"] for x in rows],marker=".",label=method)
            ax.axhline(.8,color="gray",linestyle="--",linewidth=.7)
            ax.set(title=f"{name}: {kind}",ylim=(0,1),ylabel="Pairwise ARI",xlabel="K")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    _save_figure(fig,figures/"stability-diagnostics.png")
    write_json(root/"metrics"/"feature-external.json",dict(koppen=external,valid_cells=int(valid.sum()),total_cells=len(valid),bom_quantitative_comparison="Not performed: no local reference raster. Public BOM classifications inform descriptive interpretation only."))
    write_json(root/"metrics"/"feature-display-summary.json",dict(representatives=comparison_rows))
    with (root/"metrics"/"physical-profiles.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(profiles[0]))
        writer.writeheader();writer.writerows(profiles)

    lines=["# Australian climate feature-expansion experiment","",
           f"Analysed {len(valid):,} mainland Australian and Tasmanian cells at 0.5°, 1991–2020. Derived {len(catalogue)} named candidate columns, including monthly baseline and profile representations.","",
           "## Comparison","",
           "Each representation was tested with five algorithm variants and K=3–12. Displayed results use the largest K passing the prior internal shortlist rules AND temporal ARI ≥0.80. Where none passes, the display is explicitly diagnostic, chosen by its weakest stability dimension. All resolutions and failures remain in the JSON audit. These displays are not a single winning national classification.","",
           "| Representation | Features → dimensions | Method / K | Bootstrap | Spatial | Temporal | Min area | Köppen subclass ARI | Status |",
           "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in comparison_rows:
        lines.append(f"| {r['name']} | {r['features']} → {r['dimensions']} | {r['method']} / {r['k']} | {r['bootstrap']:.3f} | {r['spatial']:.3f} | {r['temporal']:.3f} | {r['minimum_area']:.1%} | {r['subclass_ari']:.3f} | {r['status']} |")
    lines += ["","![Classification comparison](figures/features/classification-comparison.png)","",
              "## Selection and area sensitivity","",
              "The old silhouette-retention and area rules are applied separately inside each representation. Silhouettes across different feature geometries are not directly comparable. The 0% area sensitivity still requires the requested number of nonempty clusters; it is not endorsement of arbitrarily tiny classes.","",
              "| Representation | Prior-rule shortlist | Temporally confirmed | Confirmed at 0% / 0.5% / 1% / 2% area |","|---|---|---|---|"]
    for name,result in metrics["runs"].items():
        def ks(items,temporal=False):
            return ", ".join(str(x["k"]) for x in items if not temporal or x["temporal_confirmed"]) or "none"
        sensitivity=" / ".join(ks(result["area_sensitivity"][str(a)],True) for a in config["feature_expansion"]["area_sensitivity"])
        lines.append(f"| {name} | {ks(result['shortlist'])} | {ks(result['shortlist'],True)} | {sensitivity} |")
    lines += ["","## Feature-family ablations","",
              "Each ablation repeats all three stability checks at its full representation's displayed method/K. Low agreement means the family changes the partition; it does not prove that the change improves climate classification. Diagnostic references remain diagnostic.","",
              "| Representation | Omitted family | ARI to full | Bootstrap | Spatial | Temporal |","|---|---|---:|---:|---:|---:|"]
    for row in metrics["ablations"]:
        if "error" in row:
            lines.append(f"| {row['representation']} | {row['omitted_family']} | failed: {row['error']} | — | — | — |")
        else:
            lines.append(f"| {row['representation']} | {row['omitted_family']} | {row['ari_to_full']:.3f} | {row['bootstrap_stability']['mean']:.3f} | {row['spatial_stability']['mean']:.3f} | {row['temporal_stability']['mean']:.3f} |")
    failures=[(name,row) for name,result in metrics["runs"].items() for row in result["candidates"] if "error" in row]
    warned=[(name,row) for name,result in metrics["runs"].items() for row in result["candidates"] if row.get("warnings")]
    lines += ["","## Reliability and interpretation","",
              "Scaling, missing-value imputation, within-family correlation pruning, family weighting and PCA are fitted anew in each cell or spatial training sample. Temporal trials recompute physical features from five-year moving blocks and refit preprocessing. Artificial temporal joins are excluded from seasonal windows and persistence calculations. Boundary-censored dry-run estimates remain exploratory.","",
              "The benchmark uses ten cell-bootstrap trials, five spatial folds, and five temporal resamples. Non-KMeans fits use at most 2,000 area-resampled training observations. Silhouette uses a reproducible uniform subset of at most 1,500 cells; reported ARIs are unweighted cell agreement, while cluster area, scaling and centroid-label matching use geographic area weights.","",
              "Baseline runs retain all variable monthly columns and omit family normalization/pruning, preserving the old feature geometry. PCA here uses exact area-weighted covariance; the previous multiscale code used unweighted PCA. Stability fits use three initializations instead of one, so differences from the old run cannot be attributed solely to added features. Use the refreshed baseline for controlled comparisons.","",
              "Derived climatological summaries re-express existing information. Variability and persistence recover information discarded by the old 30-year averaging step. Small seasonal contrasts can still be noise; the temporal tests assess that risk. Local dry-month proxies are not standardized drought indices, and monthly mean temperatures do not measure daily heat or frost extremes.","",
              f"Candidate failures: {len(failures)}. Candidates with captured fitting warnings: {len(warned)}. All messages are saved alongside their candidate metrics. Pairwise stability means are descriptive; their dependent ARI pairs are not independent confidence-interval samples.","",
              f"Köppen comparison used {int(valid.sum()):,}/{len(valid):,} valid cells only after selection and feature artifacts were frozen. Agreement is descriptive, not an optimization target. A quantitative Bureau of Meteorology comparison was not performed because a matching local reference raster is unavailable; no new climate-data downloads were made.","",
              "## Outputs","",
              "- [Physical feature maps](figures/features/physical-feature-maps.png)",
              "- [Assignment reproducibility](figures/features/assignment-agreement.png)",
              "- [All stability curves](figures/features/stability-diagnostics.png)",
              "- [Physical profiles in original units](metrics/physical-profiles.csv)",
              "- [Complete experiment and cross-representation comparisons](metrics/feature-experiment.json)",
              "- [Named feature definitions](metrics/feature-catalogue.json)","",
              "## Reproduce","","```bash","OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 climate-unsupervised feature-all --config configs/australia-features.yaml","```","",
              "Checkpoints are reused only if the input series, source code and configuration hashes match. Run `feature-evaluate` to regenerate figures from the frozen results without refitting.",""]
    lookup={r["name"]:r for r in comparison_rows}
    if {"baseline-direct","baseline-pca","summaries-direct","variability-direct"} <= lookup.keys():
        baseline,compressed,summary,variable=[lookup[n] for n in ("baseline-direct","baseline-pca","summaries-direct","variability-direct")]
        findings=["## Main findings","",
                  f"The refreshed monthly baseline supports a displayed K={baseline['k']} without PCA, while its PCA version displays K={compressed['k']}. Thus the earlier general conclusion that all K≥5 solutions are unstable was too strong: representation and fitting/validation choices matter. This comparison does not isolate PCA as the sole cause, because it also changes the fitted geometry and the silhouette reference used for selection.","",
                  f"Physical summaries support K={summary['k']}, with temporal ARI {summary['temporal']:.3f}; compare this with the configured temporal threshold of {config['feature_expansion']['minimum_temporal_stability_ari']:.2f}. Their displayed Köppen subclass ARI is {summary['subclass_ari']:.3f}. Adding variability/persistence instead displays K={variable['k']} with subclass ARI {variable['subclass_ari']:.3f}. More feature columns have not established a clearly superior detailed national classification. External correspondence alone is not a quality verdict.","",
                  "The maps still show broad regimes. A 30-year monthly temperature/rainfall record does not contain daily heat, frost, rain-event structure or direct atmospheric moisture-demand information. The next data stage is therefore scientifically motivated by missing measurements, rather than a promise of a larger K.","",
                  "[Broader feature catalogue](../../../docs/climate-feature-catalogue.md) · [All implemented definitions](../../../docs/implemented-climate-features.md) · [Targeted data acquisition specification](../../../docs/climate-data-acquisition.md)",""]
        lines[4:4]=findings
    output=root/"feature-expansion-report.md"
    output.write_text("\n".join(lines))
    return output
