"""Build reproducible 0/+1/+2/+3 C county-winter scenario panel from saved daily-derived Daymet outputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

KEYS=['fips','year']
TEMPS=['tmin_djf_2000m','tmax_djf_2000m','tmin_mar_2000m','tmax_mar_2000m','tmin_apr_2000m','tmax_apr_2000m','tmin_nov_2000m','tmax_nov_2000m']
PRECIP=['precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m']
TERRAIN=['mean_elev_m','mean_slope_deg','mean_northness','mean_eastness','frac_above_2000m','frac_above_2500m','frac_above_3000m']

def normalize(df):
    df=df.copy()
    df['fips']=df.fips.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    df['year']=df.year.astype(int)
    return df

def build(inputs):
    train=normalize(pd.read_csv(inputs/'consistent_model_data.csv',dtype={'fips':str}))
    warm=normalize(pd.read_csv(inputs/'county_warming_scenarios_all.csv',dtype={'fips':str}))
    nov=normalize(pd.read_csv(inputs/'county_climate_panel_v4.csv',dtype={'fips':str}))
    assert len(train)==400 and train.fips.nunique()==20 and train.year.nunique()==20
    for label,d in [('train',train),('warm',warm),('nov',nov)]:
        if d.duplicated(KEYS).any():raise ValueError(f'Duplicate county-years in {label}')
    train=train.drop(columns=TEMPS[-2:],errors='ignore').merge(nov[KEYS+TEMPS[-2:]],on=KEYS,how='left',validate='one_to_one')
    if train[TEMPS].isna().any().any():raise ValueError('Missing seasonal temperature predictors')
    warm_cols=[f'{p}_{i}' for i in range(4) for p in ('pdd','fdd','thaw')]
    warm=warm[KEYS+warm_cols]
    train=train.merge(warm,on=KEYS,how='left',validate='one_to_one')
    if train[warm_cols].isna().any().any():raise ValueError('Missing daily-derived scenario values')
    for a,b in [('positive_degree_days_2000m','pdd_0'),('freezing_degree_days_2000m','fdd_0'),('thaw_days_2000m','thaw_0')]:
        if not np.allclose(train[a],train[b],atol=1e-6,rtol=0):raise ValueError(f'Baseline mismatch: {a}')
    scenarios=[]
    for delta in range(4):
        d=train.drop(columns=warm_cols).copy()
        for col in TEMPS:d[col]=d[col]+delta
        for target,prefix in [('positive_degree_days_2000m','pdd'),('freezing_degree_days_2000m','fdd'),('thaw_days_2000m','thaw')]:
            d[target]=train[f'{prefix}_{delta}'].to_numpy()
        d['warming_C']=delta
        d['pdd_squared']=d['positive_degree_days_2000m']**2
        scenarios.append(d)
    scenarios=pd.concat(scenarios,ignore_index=True)
    train=train.drop(columns=warm_cols)
    train['pdd_squared']=train['positive_degree_days_2000m']**2
    assert len(scenarios)==1600 and not scenarios.duplicated(KEYS+['warming_C']).any()
    baseline=scenarios.loc[scenarios.warming_C.eq(0)].sort_values(KEYS).reset_index(drop=True)
    original=train.sort_values(KEYS).reset_index(drop=True)
    for col in TEMPS+PRECIP+TERRAIN+['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m']:
        if not np.allclose(original[col],baseline[col],atol=1e-6,rtol=0):raise ValueError(f'Baseline differs: {col}')
    for delta in range(4):
        d=scenarios.loc[scenarios.warming_C.eq(delta)].sort_values(KEYS).reset_index(drop=True)
        for col in TEMPS:
            if not np.allclose(d[col]-baseline[col],delta,atol=1e-6):raise ValueError(f'Temperature shift: {col}')
        for col in PRECIP+TERRAIN:
            if not np.allclose(d[col],baseline[col],atol=1e-6):raise ValueError(f'Non-temperature changed: {col}')
        for col in ['positive_degree_days_2000m','thaw_days_2000m']:
            if (d[col]<baseline[col]-1e-6).any():raise ValueError(f'Unexpected decrease: {col}')
        if (d['freezing_degree_days_2000m']>baseline['freezing_degree_days_2000m']+1e-6).any():raise ValueError('FDD increased')
    if not set(original.season_days.unique()).issubset({181,182}):raise ValueError('Unexpected season length')
    return original,scenarios

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parents[1]/'inputs');ap.add_argument('--outputs',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');args=ap.parse_args()
    args.outputs.mkdir(parents=True,exist_ok=True)
    train,scenarios=build(args.inputs)
    train.to_csv(args.outputs/'train_v4.csv',index=False)
    scenarios.to_csv(args.outputs/'warming_scenarios_v4.csv',index=False)
    print('PASS: 400 historical rows, 1600 scenario rows, baseline matches, shifts/precipitation/terrain checked')
