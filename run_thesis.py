"""Reproducible final thesis analysis, using the saved historical and warming inputs.
Run: python run_thesis.py [--skip-validation] [--skip-robustness]
Requires numpy, pandas, statsmodels, scikit-learn, xgboost.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parent
IN=ROOT/'inputs'; OUT=ROOT/'outputs'
KEYS=['fips','year']; FOLDS=[(2013,2014,2016),(2016,2017,2019),(2019,2020,2023)]
TERRAIN=['mean_elev_m','mean_slope_deg','mean_northness','mean_eastness','frac_above_2000m','frac_above_2500m','frac_above_3000m']
TEMP=['tmin_nov_2000m','tmax_nov_2000m','tmin_djf_2000m','tmax_djf_2000m','tmin_mar_2000m','tmax_mar_2000m','tmin_apr_2000m','tmax_apr_2000m']
PRECIP=['precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m']
CLIMATE=TEMP+PRECIP+['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m']
ML=['tmin_djf_2000m','tmax_djf_2000m','tmin_mar_2000m','tmax_mar_2000m','tmin_apr_2000m','tmax_apr_2000m','precip_djf_2000m','precip_mar_apr_2000m','positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m']+TERRAIN+['precip_on_2000m','tmin_nov_2000m','tmax_nov_2000m']
FE={
'FE1 Baseline':['positive_degree_days_2000m']+PRECIP,
'FE2 Expanded':['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m']+PRECIP,
'FE3 Nonlinear':['positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m','pdd_squared']+PRECIP}
MODELS=list(FE)+['Random Forest','XGBoost']
TARGETS={10:'snow_days_10cm_2000m',30:'snow_days_30cm_2000m'}
BEA={'Colorado':1085.550,'Idaho':90.582,'Montana':48.993,'Utah':419.487,'Wyoming':27.216}
STATES={8:'Colorado',16:'Idaho',30:'Montana',49:'Utah',56:'Wyoming'}

def norm(d):
 d=d.copy();d['fips']=d.fips.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5);d['year']=d.year.astype(int);return d

def read(name):return norm(pd.read_csv(IN/name,dtype={'fips':str}))

def check_unique(d,keys,label):
 if d.duplicated(keys).any():raise ValueError(f'{label}: duplicate {keys}')

def check_equal(a,b,label):
 if not np.allclose(a,b,atol=1e-6,rtol=0,equal_nan=False):raise ValueError(f'{label} mismatch: max abs diff {np.max(abs(np.asarray(a)-np.asarray(b)))}')

def inputs():
 d=read('county_snodas_10cm_30cm_merged.csv')
 v=read('county_climate_panel_v4.csv')
 w=read('county_warming_scenarios_all.csv')
 for name,x in [('snow',d),('v4',v),('warming',w)]:check_unique(x,KEYS,name)
 d=d.merge(v[KEYS+TEMP[:2]],on=KEYS,how='left',validate='one_to_one')
 if len(d)!=400 or d.fips.nunique()!=20 or set(d.year)!=(set(range(2004,2024))):raise ValueError('Expected 400 county-winters, 20 counties, 2004–2023')
 if set(map(tuple,d[KEYS].to_numpy()))!=set(map(tuple,w[KEYS].to_numpy())):raise ValueError('Warming keys differ from modeling sample')
 d=d.merge(w[KEYS+[f'{p}_{k}' for p in ['pdd','fdd','thaw'] for k in range(4)]],on=KEYS,validate='one_to_one')
 for old,new in [('positive_degree_days_2000m','pdd_0'),('freezing_degree_days_2000m','fdd_0'),('thaw_days_2000m','thaw_0')]:check_equal(d[old],d[new],old+' baseline')
 required=list(dict.fromkeys(list(TARGETS.values())+CLIMATE+TERRAIN+['season_days','valid_fraction_2000m']))
 if d[required].isna().any().any():raise ValueError('Missing final modeling input')
 if len(ML)!=21:raise ValueError('ML should have 21 predictors')
 frames=[]
 for k in range(4):
  x=d.copy();x['warming_C']=k
  for t in TEMP:x[t]+=k
  for old,prefix in [('positive_degree_days_2000m','pdd'),('freezing_degree_days_2000m','fdd'),('thaw_days_2000m','thaw')]:x[old]=x[f'{prefix}_{k}']
  x['pdd_squared']=x.positive_degree_days_2000m**2
  frames.append(x)
 scenarios=pd.concat(frames,ignore_index=True)
 d['pdd_squared']=d.positive_degree_days_2000m**2
 check_unique(scenarios,KEYS+['warming_C'],'scenario')
 if len(scenarios)!=1600:raise ValueError('Expected 1600 scenario observations')
 base=scenarios[scenarios.warming_C.eq(0)].sort_values(KEYS).reset_index(drop=True)
 ordered=d.sort_values(KEYS).reset_index(drop=True)
 for f in ML+list(dict.fromkeys(sum(FE.values(),[]))):check_equal(base[f],ordered[f],f)
 for f in PRECIP+TERRAIN:
  pivot=scenarios.pivot(index=KEYS,columns='warming_C',values=f)
  for k in (1,2,3):check_equal(pivot[k],pivot[0],f+' held fixed')
 for f in TEMP:
  pivot=scenarios.pivot(index=KEYS,columns='warming_C',values=f)
  for k in (1,2,3):check_equal(pivot[k]-pivot[0],np.full(len(pivot),k),f+' warming shift')
 # Preserve original input order; sorting can alter subsampled XGBoost fits.
 return d,scenarios

def estimator(name):
 if name=='Random Forest':return RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.7,random_state=42,n_jobs=-1)
 return XGBRegressor(n_estimators=300,max_depth=2,learning_rate=.03,subsample=.8,colsample_bytree=.8,min_child_weight=5,reg_lambda=10,objective='reg:squarederror',random_state=42,n_jobs=-1)

def fit_predict(name,train,test,target,mode):
 if name in FE:
  formula=target+' ~ '+' + '.join(FE[name])+(' + C(fips)' if mode in ('LOYO','Forward') else '')
  return smf.ols(formula,data=train).fit().predict(test).to_numpy()
 model=estimator(name);model.fit(train[ML],train[target]);return model.predict(test[ML])

def score(y,p):return {'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p))}

def validate(d,target,mode):
 rows=[];predictions=[]
 for name in MODELS:
  pred=np.full(len(d),np.nan)
  if mode=='Forward':
   folds=[(d.year.le(end).to_numpy(),d.year.between(start,stop).to_numpy()) for end,start,stop in FOLDS]
  else:
   group='year' if mode=='LOYO' else 'fips'
   folds=[(d[group].ne(held).to_numpy(),d[group].eq(held).to_numpy()) for held in sorted(d[group].unique())]
  for tr,te in folds:pred[te]=fit_predict(name,d.loc[tr],d.loc[te],target,mode)
  valid=np.isfinite(pred)
  rows.append({'threshold_cm':10 if '10cm' in target else 30,'validation':mode,'Model':name,'n_obs_scored':int(valid.sum()),**score(d.loc[valid,target],pred[valid])})
  predictions.append(pd.DataFrame({'fips':d.loc[valid,'fips'],'year':d.loc[valid,'year'],'threshold_cm':rows[-1]['threshold_cm'],'validation':mode,'Model':name,'observed':d.loc[valid,target],'predicted':pred[valid]}))
 return rows,predictions

def models_and_scenarios(d,s,threshold):
 target=TARGETS[threshold]; out=OUT/f'snodas_{threshold}cm_v4_november';out.mkdir(parents=True,exist_ok=True)
 pred=s[KEYS+['warming_C','state','county']].copy()
 for name in MODELS:
  if name in FE:
   fit=smf.ols(target+' ~ '+' + '.join(FE[name])+' + C(fips) + C(year)',data=d).fit(cov_type='cluster',cov_kwds={'groups':d.fips})
   raw=fit.predict(s).to_numpy()
   with open(out/'fixed_effects_coefficients.txt','a' if name!=MODELS[0] else 'w') as f:f.write('\n'+name+'\n'+fit.summary().as_text()+'\n')
  else:
   fit=estimator(name);fit.fit(d[ML],d[target]);raw=fit.predict(s[ML])
  pred[name]=np.clip(raw,0,s.season_days.to_numpy())
 baseline=pred.loc[pred.warming_C.eq(0),KEYS+MODELS].rename(columns={n:n+'_baseline' for n in MODELS})
 pred=pred.merge(baseline,on=KEYS,validate='many_to_one')
 for n in MODELS:
  pred[n+'_change']=pred[n]-pred[n+'_baseline']
  pred[n+'_pct_change']=100*pred[n+'_change']/pred[n+'_baseline'].replace(0,np.nan)
 pred.to_csv(out/'snodas_warming_predictions.csv',index=False)
 summary=pred.groupby('warming_C')[[n+'_change' for n in MODELS]].mean().reset_index();summary.columns=['warming_C']+MODELS
 summary.to_csv(out/'snodas_warming_day_changes.csv',index=False)
 state=pd.concat([pred[['state','warming_C',n,n+'_baseline']].rename(columns={n:'predicted_days',n+'_baseline':'baseline_days'}).assign(Model=n) for n in MODELS],ignore_index=True)
 state=state.groupby(['state','warming_C','Model'],as_index=False)[['predicted_days','baseline_days']].mean()
 state['pct_change']=100*(state.predicted_days/state.baseline_days-1)
 state.to_csv(out/'snodas_state_warming_summary.csv',index=False)
 print('\n',threshold,'cm mean change in days\n',summary.round(3).to_string(index=False))
 return state

def economic(states):
 rows=[]
 for threshold,state in states.items():
  state=state.copy();state['state_name']=state.state.astype(int).map(STATES)
  if set(state.state_name)!=set(BEA):raise ValueError('Unexpected state codes')
  if state.duplicated(['state','warming_C','Model']).any():raise ValueError('Duplicate state/model/warming')
  for r in state.itertuples():
   for eta in (.25,.5,1.):
    va=BEA[r.state_name];delta=va*eta*(r.predicted_days/r.baseline_days-1)
    rows.append({'threshold':f'{threshold}cm','state':r.state_name,'state_fips':int(r.state),'Model':r.Model,'warming_C':int(r.warming_C),'elasticity':eta,'bea_2023_value_added_M':va,'predicted_days':r.predicted_days,'baseline_days':r.baseline_days,'snow_change_pct':r.pct_change,'change_value_added_M':delta,'implied_value_added_M':va+delta})
 detail=pd.DataFrame(rows)
 if len(detail)!=600:raise ValueError('Expected 600 economic detail rows')
 out=OUT/'snodas_economic_analysis_v4';out.mkdir(parents=True,exist_ok=True)
 detail.to_csv(out/'economic_exposure_by_state.csv',index=False)
 total=detail.groupby(['threshold','Model','warming_C','elasticity'],as_index=False).agg(change_value_added_M=('change_value_added_M','sum'),implied_value_added_M=('implied_value_added_M','sum'),bea_2023_value_added_M=('bea_2023_value_added_M','sum'))
 total['change_pct_of_2023_VA']=100*total.change_value_added_M/total.bea_2023_value_added_M
 total.to_csv(out/'economic_exposure_five_state_totals.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--skip-validation',action='store_true');p.add_argument('--skip-robustness',action='store_true');args=p.parse_args()
 OUT.mkdir(exist_ok=True)
 d,s=inputs()
 d[KEYS+list(TARGETS.values())+['valid_fraction_2000m']].to_csv(OUT/'final_sample_snow_summary.csv',index=False)
 print('Input checks PASS: 400 county-winters, 21 ML features, 1600 scenario rows, baseline and fixed-feature checks')
 print('Observed means:',{k:round(d[v].mean(),4) for k,v in TARGETS.items()})
 states={k:models_and_scenarios(d,s,k) for k in TARGETS}
 economic(states)
 if not args.skip_validation:
  rows=[];pred=[]
  for k,t in TARGETS.items():
   for mode in ['LOYO','Forward']:
    a,b=validate(d,t,mode);rows+=a;pred+=b
  pd.DataFrame(rows).to_csv(OUT/'model_validation.csv',index=False)
  pd.concat(pred,ignore_index=True).to_csv(OUT/'model_validation_predictions.csv',index=False)
 if not args.skip_robustness:
  filtered=d[d.valid_fraction_2000m.ge(.95)].reset_index(drop=True)
  if len(filtered)!=392:raise ValueError('Expected 392 rows after low-coverage exclusion')
  rows=[];pred=[]
  for k,t in TARGETS.items():
   for mode,frame,label in [('LOYO',d,'LOYO'),('LOYO',filtered,'LOYO_low_coverage_excluded'),('LOCO',d,'LOCO')]:
    a,b=validate(frame,t,mode)
    for r in a:r['validation']=label
    for x in b:x['validation']=label
    rows+=a;pred+=b
  out=OUT/'snodas_remaining_robustness_v4';out.mkdir(exist_ok=True)
  pd.DataFrame(rows).to_csv(out/'robustness_validation_results.csv',index=False)
  pd.concat(pred,ignore_index=True).to_csv(out/'robustness_oof_predictions.csv',index=False)
 print('Finished. Outputs:',OUT)
if __name__=='__main__':main()
