import json

import numpy as np

from climate_unsupervised import config as config_module
from climate_unsupervised import feature_experiment as experiment
from climate_unsupervised.artifacts import verify_manifest_inputs
from climate_unsupervised.config import load_config
from climate_unsupervised.feature_engineering import FeatureTable, FeatureTransformer, derive_features


def test_resample_preprocessing_is_fitted_only_on_training_cells():
    config=load_config(config_module.PROJECT_ROOT/'configs/australia-features.yaml')
    config['multiscale'].update(seeds=[42,43,44],spatial_folds=3,algorithm_sample_max=20)
    config['feature_expansion']['temporal_seeds']=[72,73,74]
    rng=np.random.default_rng(3)
    x=rng.normal(size=(36,4))
    x[-1]+=100  # A held-out outlier must not shift a training mean.
    meta=[dict(name=str(i),family=str(i),transform='identity') for i in range(4)]
    table=FeatureTable(x,meta)
    weights=np.linspace(.7,1,36)
    lat=np.repeat(np.arange(-40,-28,2),6)
    lon=np.tile(np.arange(120,132,2),6)
    _,_,trials=experiment.make_trials(table,[table]*3,weights,lat,lon,config,False,False)
    for kind in ('bootstrap','spatial'):
        for transformed,pool,resampled,_,_ in trials[kind]:
            train_weights=np.ones(len(pool)) if resampled else weights[pool]
            assert np.allclose(np.average(transformed[pool],axis=0,weights=train_weights),0,atol=1e-10)


def test_small_feature_benchmark_freezes_and_resumes(tmp_path,monkeypatch):
    config=load_config(config_module.PROJECT_ROOT/'configs/australia-features.yaml')
    config['run']['output_root']='run'
    config['multiscale'].update(k_min=3,k_max=3,seeds=[42,43,44],spatial_folds=3,algorithm_sample_max=80,final_n_init=2)
    config['feature_expansion'].update(representations=['profiles'],encodings=['direct'],temporal_seeds=[72,73,74],temporal_block_years=3,stability_n_init=2,silhouette_sample_max=80)
    monkeypatch.setattr(config_module,'PROJECT_ROOT',tmp_path)
    rng=np.random.default_rng(8)
    years=np.arange(1991,2003)
    month=np.arange(12)[None,:,None]
    group=np.repeat([0,1,2],30)[None,None,:]
    t=12+group*8+(2+group)*np.cos(2*np.pi*month/12)+rng.normal(size=(12,12,90))
    p=np.exp(3+group*.5+.5*np.cos(2*np.pi*month/12)+rng.normal(scale=.2,size=(12,12,90)))
    coords=dict(area_weights=np.ones(90),latitude=np.repeat(np.arange(-40,-22,2),10),longitude=np.tile(np.arange(120,140,2),9),cell_id=np.array([str(i) for i in range(90)]))
    processed=tmp_path/'run/data/processed'; processed.mkdir(parents=True)
    metric_dir=tmp_path/'run/results/metrics';metric_dir.mkdir(parents=True)
    manifest_dir=tmp_path/'run/results/manifests';manifest_dir.mkdir(parents=True)
    series=processed/'monthly_series.npz'
    np.savez_compressed(series,temperature_c=t,precipitation_mm=p,years=years,**coords)
    table=derive_features(t,p,years)
    np.savez_compressed(processed/'expanded_features.npz',features=table.values,feature_names=np.array([m['name'] for m in table.metadata]),**coords)
    (metric_dir/'feature-catalogue.json').write_text(json.dumps(dict(count=len(table.metadata),features=table.metadata)))
    (manifest_dir/'feature-prepare.json').write_text('{}')
    monkeypatch.setattr(experiment,'prepare_feature_series',lambda *args:series)
    output=experiment.run_feature_experiment(config)
    payload=json.loads(output.read_text())
    rows=payload['runs']['profiles-direct']['candidates']
    assert len(rows)==5
    assert all('temporal_stability' in r for r in rows)
    assert payload['external_labels_used'] is False
    frozen=json.loads((manifest_dir/'feature-frozen.json').read_text())
    verify_manifest_inputs(frozen)
    # Resume must not fit any models again.
    monkeypatch.setattr(experiment,'fit_candidate',lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected refit')))
    experiment.run_feature_experiment(config)
    assert json.loads(output.read_text())['runs']==payload['runs']
    # Tampering is detected before external evaluation.
    output.write_text(output.read_text()+' ')
    import pytest
    with pytest.raises(RuntimeError,match='Frozen input'):
        verify_manifest_inputs(frozen)
