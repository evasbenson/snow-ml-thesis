# Run in Jupyter with %run -i ~/Files/thesis/snow_ml/code/run_snodas_comparison.py
# Loads saved training data directly; requires warming_model_df from the notebook.
# Requires county_climate_panel_v4.csv in ~/Files/thesis/snow_ml/ml_model/.
# Run from Jupyter: %run -i ~/Files/thesis/snow_ml/code/run_snodas_v4_november.py
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = Path.home() / 'Files/thesis/snow_ml/ml_model'
OUT = BASE / 'snodas_10cm_v4_november'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 'snow_days_10cm_2000m'
KEYS = ['fips','year']
SCENARIO_KEYS = KEYS + ['warming_C']
# Reimport the saved 400-row modeling dataset rather than relying on notebook `data`.
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

fe_specs = {
    'FE1 Baseline': ['positive_degree_days_2000m', 'precip_on_2000m', 'precip_djf_2000m', 'precip_mar_apr_2000m'],
    'FE2 Expanded': ['positive_degree_days_2000m', 'freezing_degree_days_2000m', 'thaw_days_2000m', 'precip_on_2000m', 'precip_djf_2000m', 'precip_mar_apr_2000m'],
    'FE3 Nonlinear': ['positive_degree_days_2000m', 'freezing_degree_days_2000m', 'thaw_days_2000m', 'pdd_squared', 'precip_on_2000m', 'precip_djf_2000m', 'precip_mar_apr_2000m'],
}
FOLDS = [(2013,2014,2016),(2016,2017,2019),(2019,2020,2023)]
terrain_cols = ['mean_elev_m','mean_slope_deg','mean_northness','mean_eastness','frac_above_2000m','frac_above_2500m','frac_above_3000m']
ml_updated_features = ['tmin_djf_2000m','tmax_djf_2000m','tmin_mar_2000m','tmax_mar_2000m','tmin_apr_2000m','tmax_apr_2000m','precip_djf_2000m','precip_mar_apr_2000m','positive_degree_days_2000m','freezing_degree_days_2000m','thaw_days_2000m'] + terrain_cols + ['precip_on_2000m']

def make_ml_model(kind):
    if kind == 'Random Forest':
        return RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.7,random_state=42,n_jobs=-1)
    if kind == 'XGBoost':
        return XGBRegressor(n_estimators=300,max_depth=2,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=10,objective='reg:squarederror',random_state=42,n_jobs=-1)
    raise ValueError(kind)

models = list(fe_specs) + ['Random Forest','XGBoost']
train_path = BASE / 'consistent_model_comparison' / 'consistent_model_data.csv'
if not train_path.exists():
    raise FileNotFoundError(f'Run the THESIS MODEL UPDATE cell once to create {train_path}')
train = pd.read_csv(train_path, dtype={'fips':str})
# The warming scenarios are constructed earlier in the notebook and not saved by cell 270.
if 'warming_model_df' not in globals():
    raise NameError('warming_model_df is not loaded. Run the notebook cell that creates warming_model_df first.')
scenarios = warming_model_df.copy()
print('Loaded training data:',train_path,'rows:',len(train))
for df in (train, scenarios):
    df['fips'] = df['fips'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    df['year'] = df['year'].astype(int)
scenarios['warming_C'] = pd.to_numeric(scenarios['warming_C'])

# Require Daymet V4: this run must not silently fall back to old features.
nov_features = ['tmin_nov_2000m', 'tmax_nov_2000m']
v4 = BASE / 'county_climate_panel_v4.csv'
ml_features = list(ml_updated_features)
if not v4.exists():
    raise FileNotFoundError(f'Copy the uploaded county_climate_panel_v4.csv to {v4}')
if v4.exists():
    nov = pd.read_csv(v4, dtype={'fips':str})
    nov['fips'] = nov['fips'].str.zfill(5)
    nov['year'] = nov['year'].astype(int)
    assert not nov.duplicated(KEYS).any()
    nov = nov[KEYS + nov_features]
    if nov[nov_features].isna().all(axis=1).any():
        print('Note: V4 contains missing high-elevation climate values outside the final sample; validating the 400 modeling rows below.')
    # Remove stale November columns if notebook variables already contain them.
    train = train.drop(columns=nov_features, errors='ignore').merge(nov, on=KEYS, validate='one_to_one')
    scenarios = scenarios.drop(columns=nov_features, errors='ignore').merge(nov, on=KEYS, validate='many_to_one')
    for feature in nov_features:
        scenarios[feature] = scenarios[feature] + scenarios['warming_C']
    ml_features = list(dict.fromkeys(ml_features + nov_features))
    print('Using v4 November minimum and maximum temperatures in both ML models.')


train['pdd_squared'] = train['positive_degree_days_2000m'] ** 2
scenarios['pdd_squared'] = scenarios['positive_degree_days_2000m'] ** 2
print(f'ML predictors: {len(ml_features)} ({len(ml_updated_features)} existing + {len(nov_features)} November)')
assert len(train) == 400 and train['fips'].nunique() == 20
assert len(scenarios) == 1600 and set(scenarios['warming_C']) == {0,1,2,3}
assert not train.duplicated(KEYS).any()
assert not scenarios.duplicated(SCENARIO_KEYS).any()
assert set(map(tuple,train[KEYS].to_numpy())) == set(map(tuple,scenarios.loc[scenarios.warming_C.eq(0),KEYS].to_numpy()))
fe_features = list(dict.fromkeys(f for fs in fe_specs.values() for f in fs))
required = list(dict.fromkeys([TARGET] + fe_features + ml_features))
for label, frame, cols in [('train',train,required),('scenarios',scenarios,fe_features+ml_features+['season_days'])]:
    missing = [c for c in cols if c not in frame]
    if missing: raise ValueError(f'{label} missing columns: {missing}')
    if frame[cols].isna().any().any(): raise ValueError(f'{label} has missing values in required columns')

# Verify all historical scenario predictors equal training predictors.
base = scenarios.loc[scenarios.warming_C.eq(0),KEYS+list(dict.fromkeys(fe_features+ml_features))]
check = train[KEYS+list(dict.fromkeys(fe_features+ml_features))].merge(base,on=KEYS,validate='one_to_one',suffixes=('_train','_scenario'))
for feature in dict.fromkeys(fe_features+ml_features):
    if not np.allclose(check[f'{feature}_train'],check[f'{feature}_scenario'],atol=1e-6,rtol=0):
        raise ValueError(f'Historical baseline mismatch for {feature}')

# Confirm precipitation remains fixed under warming.
for feature in ['precip_on_2000m','precip_djf_2000m','precip_mar_apr_2000m']:
    reference = base[KEYS+[feature]].rename(columns={feature:'reference'})
    test = scenarios[KEYS+[feature]].merge(reference,on=KEYS,validate='many_to_one')
    if not np.allclose(test[feature],test['reference'],atol=1e-6,rtol=0):
        raise ValueError(f'Precipitation changes under warming: {feature}')

train = train.sort_values(KEYS).reset_index(drop=True)

def score(y,p):
    return {'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p))}

# Match original validation: county FE, without year FE, for held-out years.
results = []
for name in models:
    y = train[TARGET]
    predictions_loyo = np.full(len(train),np.nan)
    predictions_forward = np.full(len(train),np.nan)
    if name in fe_specs:
        formula = TARGET + ' ~ ' + ' + '.join(fe_specs[name]) + ' + C(fips)'
        def fit_predict(tr,te):
            fit = smf.ols(formula,data=train.loc[tr]).fit()
            return fit.predict(train.loc[te])
    else:
        def fit_predict(tr,te):
            model = make_ml_model(name)
            model.fit(train.loc[tr,ml_features],train.loc[tr,TARGET])
            return model.predict(train.loc[te,ml_features])
    for yr in sorted(train.year.unique()):
        tr = train.year.ne(yr)
        te = ~tr
        predictions_loyo[te.to_numpy()] = fit_predict(tr,te)
    for train_end,test_start,test_end in FOLDS:
        tr = train.year.le(train_end)
        te = train.year.between(test_start,test_end)
        predictions_forward[te.to_numpy()] = fit_predict(tr,te)
    valid = np.isfinite(predictions_forward)
    a = score(y,predictions_loyo)
    b = score(y[valid],predictions_forward[valid])
    results.append({'Model':name, **{f'LOYO {k}':v for k,v in a.items()},**{f'Forward {k}':v for k,v in b.items()}})
    print('Validated:',name)
validation = pd.DataFrame(results)
validation.to_csv(OUT/'snodas_validation.csv',index=False)
print('\nSNODAS VALIDATION')
print(validation.round(3).to_string(index=False))

# Match original final models: county AND year FE; cluster by county.
pred = scenarios[SCENARIO_KEYS].copy()
for optional in ['state','county']:
    if optional in scenarios: pred[optional] = scenarios[optional].to_numpy()
for name in models:
    if name in fe_specs:
        formula = TARGET + ' ~ ' + ' + '.join(fe_specs[name]) + ' + C(fips) + C(year)'
        fit = smf.ols(formula,data=train).fit(cov_type='cluster',cov_kwds={'groups':train.fips})
        values = fit.predict(scenarios)
    else:
        fit = make_ml_model(name)
        fit.fit(train[ml_features],train[TARGET])
        values = fit.predict(scenarios[ml_features])
    pred[name] = np.clip(np.asarray(values),0,scenarios['season_days'].to_numpy())

baseline = pred.loc[pred.warming_C.eq(0),KEYS+models].rename(columns={name:f'{name}_baseline' for name in models})
pred = pred.merge(baseline,on=KEYS,validate='many_to_one')
for name in models:
    pred[f'{name}_change'] = pred[name]-pred[f'{name}_baseline']
    pred[f'{name}_pct_change'] = 100*pred[f'{name}_change']/pred[f'{name}_baseline'].replace(0,np.nan)
summary = pred.groupby('warming_C')[[f'{name}_change' for name in models]].mean().reset_index()
summary.columns = ['warming_C']+models
pred.to_csv(OUT/'snodas_warming_predictions.csv',index=False)
summary.to_csv(OUT/'snodas_warming_day_changes.csv',index=False)
print('\nSNODAS MEAN CHANGE IN SNOW DAYS')
print(summary.round(3).to_string(index=False))

# State summary for economic valuation: ratio of mean predicted change to mean baseline.
if 'state' in pred:
    long = pd.concat([
        pred[KEYS+['state','warming_C',name,f'{name}_baseline']].rename(columns={name:'predicted_days',f'{name}_baseline':'baseline_days'}).assign(Model=name)
        for name in models
    ],ignore_index=True)
    state = long.groupby(['state','warming_C','Model'],as_index=False)[['predicted_days','baseline_days']].mean()
    state['pct_change'] = 100*(state['predicted_days']-state['baseline_days'])/state['baseline_days'].replace(0,np.nan)
    state.to_csv(OUT/'snodas_state_warming_summary.csv',index=False)
    print('\nSaved state-level percentage changes for BEA ORSA calculations.')
print('\nAll outputs saved in:',OUT)
