"""Remaining SNODAS robustness checks: coverage-filter LOYO and LOCO.
Run: python run_snodas_remaining_robustness.py
Optional: python run_snodas_remaining_robustness.py --data-dir /path/to/ml_model --output-dir /path/to/output
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,default=Path.home()/'Files/thesis/snow_ml/ml_model')
parser.add_argument('--output-dir',type=Path,default=None)
args=parser.parse_args()
base=args.data_dir
out=args.output_dir or base/'snodas_remaining_robustness_v4'
out.mkdir(parents=True,exist_ok=True)

# The saved 400-row dataset used by the final model comparisons.
train_path=base/'consistent_model_comparison/consistent_model_data.csv'
if train_path.exists():
    df=pd.read_csv(train_path,dtype={'fips':str})
else:
    fallback=base/'county_snodas_10cm_30cm_merged.csv'
    if not fallback.exists():
        raise FileNotFoundError(f'Missing {train_path} and {fallback}')
    df=pd.read_csv(fallback,dtype={'fips':str})

nov=pd.read_csv(base/'county_climate_panel_v4.csv',dtype={'fips':str})
for x in (df,nov):
    x['fips']=x['fips'].str.replace(r'\.0$','',regex=True).str.zfill(5)
    x['year']=x['year'].astype(int)
keys=['fips','year']
nov_cols=['tmin_nov_2000m','tmax_nov_2000m']
assert not nov.duplicated(keys).any()
df=df.drop(columns=nov_cols,errors='ignore').merge(nov[keys+nov_cols],on=keys,validate='one_to_one')
df['pdd_squared']=df['positive_degree_days_2000m']**2
if 'snow_days_30cm_2000m' not in df:
    snow=pd.read_csv(base/'county_snodas_10cm_30cm_merged.csv',dtype={'fips':str})
    snow['fips']=snow['fips'].str.zfill(5)
    df=df.merge(snow[keys+['snow_days_30cm_2000m','valid_fraction_2000m']],on=keys,validate='one_to_one')
if 'valid_fraction_2000m' not in df:
    snow=pd.read_csv(base/'county_snodas_10cm_30cm_merged.csv',dtype={'fips':str})
    snow['fips']=snow['fips'].str.zfill(5)
    df=df.merge(snow[keys+['valid_fraction_2000m']],on=keys,validate='one_to_one')

fe_specs={
'FE1 Baseline':['positive_degree_days_2000m','precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m'],
'FE2 Expanded':['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m','precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m'],
'FE3 Nonlinear':['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m','pdd_squared','precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m']}
terrain=['mean_elev_m','mean_slope_deg','mean_northness','mean_eastness','frac_above_2000m','frac_above_2500m','frac_above_3000m']
ml_features=['tmin_djf_2000m','tmax_djf_2000m','tmin_mar_2000m','tmax_mar_2000m','tmin_apr_2000m','tmax_apr_2000m','precip_djf_2000m','precip_mar_apr_2000m','positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m']+terrain+['precip_on_2000m']+nov_cols
models=list(fe_specs)+['Random Forest','XGBoost']
required=['snow_days_10cm_2000m','snow_days_30cm_2000m','valid_fraction_2000m']+list(dict.fromkeys(sum(fe_specs.values(),[])+ml_features))
assert len(df)==400 and df.fips.nunique()==20 and df.year.nunique()==20, f'Unexpected sample: {df.shape}'
assert not df.duplicated(keys).any()
assert not df[required].isna().any().any(), 'Missing modeling values'
assert len(ml_features)==21

def estimator(name):
    if name=='Random Forest':
        return RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.7,random_state=42,n_jobs=-1)
    return XGBRegressor(n_estimators=300,max_depth=2,learning_rate=.03,subsample=.8,colsample_bytree=.8,min_child_weight=5,reg_lambda=10,objective='reg:squarederror',random_state=42,n_jobs=-1)

def metrics(y,p):
    return {'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p))}

def validate(data,target,kind):
    # LOYO: county indicators allowed (all counties present in each training fold).
    # LOCO: county indicators prohibited (held-out county has no fitted county effect).
    rows=[];pred_rows=[]
    group='year' if kind=='LOYO' else 'fips'
    for name in models:
        p=np.full(len(data),np.nan)
        for held in sorted(data[group].unique()):
            tr=data[group].ne(held);te=~tr
            a=data.loc[tr];b=data.loc[te]
            if name in fe_specs:
                formula=target+' ~ '+' + '.join(fe_specs[name])+(' + C(fips)' if kind=='LOYO' else '')
                fitted=smf.ols(formula,data=a).fit()
                predicted=fitted.predict(b)
            else:
                fitted=estimator(name)
                fitted.fit(a[ml_features],a[target])
                predicted=fitted.predict(b[ml_features])
            p[te.to_numpy()]=np.asarray(predicted)
        assert np.isfinite(p).all()
        rows.append({'threshold_cm':10 if '10cm' in target else 30,'validation':kind,'model':name,
                     'model_type':'pooled linear (no county FE)' if kind=='LOCO' and name in fe_specs else ('county FE' if name in fe_specs else 'ML'),
                     'n_obs':len(data),'n_counties':data.fips.nunique(),**metrics(data[target],p)})
        pred_rows.append(pd.DataFrame({'fips':data.fips,'year':data.year,'threshold_cm':10 if '10cm' in target else 30,'validation':kind,'model':name,'observed':data[target],'predicted':p}))
        print(f'{target} {kind} {name}: R2={rows[-1]["R2"]:.3f}, MAE={rows[-1]["MAE"]:.2f}, RMSE={rows[-1]["RMSE"]:.2f}',flush=True)
    return rows,pred_rows

print('Full sample:',len(df),'counties:',df.fips.nunique(),flush=True)
low=df.valid_fraction_2000m.lt(.95)
print('Low coverage (<95%):',low.sum(),'observations',flush=True)
assert low.sum()==8, f'Expected 8 low-coverage observations; found {low.sum()}'
filtered=df.loc[~low].reset_index(drop=True)
print('Filtered sample:',len(filtered),flush=True)
all_rows=[];all_preds=[]
for target in ['snow_days_10cm_2000m','snow_days_30cm_2000m']:
    for kind,data in [('LOYO',df.reset_index(drop=True)),('LOYO_low_coverage_excluded',filtered),('LOCO',df.reset_index(drop=True))]:
        # reuse LOYO logic with descriptive label
        actual_kind='LOYO' if kind.startswith('LOYO') else kind
        rows,preds=validate(data,target,actual_kind)
        for r in rows:r['validation']=kind
        for p in preds:p['validation']=kind
        all_rows.extend(rows);all_preds.extend(preds)
results=pd.DataFrame(all_rows)
results.to_csv(out/'robustness_validation_results.csv',index=False)
pd.concat(all_preds,ignore_index=True).to_csv(out/'robustness_oof_predictions.csv',index=False)
print('\nSaved:',out/'robustness_validation_results.csv',flush=True)
print('Saved:',out/'robustness_oof_predictions.csv',flush=True)
