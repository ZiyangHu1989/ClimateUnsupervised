import calendar

import numpy as np
import pytest

from climate_unsupervised.feature_engineering import (
    FeatureTable, FeatureTransformer, circular_quarters, derive_features, _windows,
)
from climate_unsupervised.feature_experiment import resample_years, _agreement_to_reference, make_trials


def series():
    years=np.arange(1991,2021)
    month=np.arange(12)[None,:,None]
    cells=np.arange(8)[None,None,:]
    t=np.broadcast_to(20+cells+5*np.cos(2*np.pi*month/12),(30,12,8)).copy()
    p=np.broadcast_to(50+cells+20*np.cos(2*np.pi*month/12),(30,12,8)).copy()
    return t,p,years


def column(table,name):
    return table.values[:,[m['name'] for m in table.metadata].index(name)]


def test_circular_quarter_and_harmonics():
    t,p,years=series()
    table=derive_features(t,p,years)
    assert np.allclose(column(table,'temperature_h1_cos'),5)
    assert np.allclose(column(table,'temperature_h1_sin'),0,atol=1e-12)
    assert np.allclose(column(table,'temperature_h2_cos'),0,atol=1e-12)
    assert np.allclose(column(table,'rainfall_mean'),p[0].sum(axis=0))
    assert np.allclose(circular_quarters(p[0])[11],p[0,[11,0,1]].sum(axis=0))
    assert np.allclose(column(table,'temperature_warmest_quarter'),t[0,[11,0,1]].mean(axis=0))
    assert np.allclose(column(table,'annual_rainfall_cv'),0,atol=1e-12)
    assert len({m['name'] for m in table.metadata})==table.values.shape[1]


def test_day_weighted_temperature_and_interannual_variability():
    t,p,years=series()
    p*=np.linspace(.5,1.5,len(years))[:,None,None]
    table=derive_features(t,p,years)
    assert np.allclose(column(table,'annual_rainfall_cv'),np.std(np.linspace(.5,1.5,30),ddof=1))
    days=np.array([[calendar.monthrange(int(y),m)[1] for m in range(1,13)] for y in years])
    expected=((t*days[:,:,None]).sum(axis=1)/days.sum(axis=1)[:,None]).mean(axis=0)
    assert np.allclose(column(table,'temperature_mean'),expected)


def test_constant_zero_rainfall_and_missing_input():
    t,p,years=series()
    p[:,:,0]=0
    t[:,:,0]=20
    table=derive_features(t,p,years)
    assert np.isnan(column(table,'rainfall_entropy')[0])
    assert np.isnan(column(table,'temperature_rainfall_correlation')[0])
    chosen=table.select('variability')
    transformer=FeatureTransformer(True).fit(chosen,np.ones(8))
    assert np.isfinite(transformer.transform(chosen)).all()
    p[0,0,0]=np.nan
    with pytest.raises(ValueError,match='complete'):
        derive_features(t,p,years)


def test_temporal_resampling_never_connects_artificial_years():
    years=np.array([2001,2002,2010])
    values=np.ones((3,12,1))
    windows,months=_windows(values,years,3)
    assert len(windows)==32  # 22 contiguous windows plus 10 within the isolated year
    assert (months==11).sum()==1
    assert np.all(windows==3)
    assert np.array_equal(resample_years(np.arange(30),42,5),resample_years(np.arange(30),42,5))


def test_training_only_imputation_pruning_and_family_balance():
    meta=[dict(name=n,family=f,transform='identity') for n,f in [('a','x'),('duplicate','x'),('b','y')]]
    train=FeatureTable(np.array([[0,0,-1],[1,2,1],[2,4,-1],[3,6,1.]]),meta)
    tr=FeatureTransformer().fit(train,np.array([1,2,3,4.]))
    assert tr.audit['retained_names']==['a','b']
    z=tr.transform(train)
    assert np.allclose(np.average(z*z,axis=0,weights=[1,2,3,4]),1)
    heldout=FeatureTable(np.array([[np.nan,1000,1000.]]),meta)
    before=tr.mean.copy()
    assert tr.transform(heldout)[0,0]==0
    assert np.array_equal(before,tr.mean)
    tr2=FeatureTransformer(correlation_threshold=1.0).fit(train,np.ones(4))
    assert tr2.audit['retained_features']==3
    for f in ['x','y']:
        cols=[i for i,j in enumerate(tr2.keep) if meta[j]['family']==f]
        assert np.isclose(np.mean(tr2.transform(train)[:,cols]**2,axis=0).sum(),1)


def test_area_weighted_pca_covariance_and_schema():
    table=FeatureTable(np.array([[0,1],[2,0],[1,4.]]),[dict(name=n,family=n,transform='identity') for n in ['a','b']])
    tr=FeatureTransformer(True).fit(table,np.array([1,2,4.]))
    z=tr.transform(table)
    assert np.allclose(np.average(z,axis=0,weights=[1,2,4]),0,atol=1e-12)
    with pytest.raises(ValueError,match='schema'):
        tr.transform(FeatureTable(table.values,table.metadata[::-1]))


def test_permutation_matched_cell_agreement():
    a=np.array([0,0,1,1,2,2])
    b=np.array([2,2,0,0,1,1])
    assert _agreement_to_reference(a,b,np.ones(6)).all()
