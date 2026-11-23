"""Focused XGBoost reproducibility audit for the ski thesis.
Place this file next to run_thesis.py and run: python3 verify_xgboost.py
No files are overwritten; results saved in outputs/xgboost_diagnostic/.
"""
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import xgboost
import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import run_thesis as thesis

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'xgboost_diagnostic'
OUT.mkdir(parents=True, exist_ok=True)
print('Python:', sys.version.split()[0], 'Platform:', platform.platform())
print('XGBoost:', xgboost.__version__, 'scikit-learn:', sklearn.__version__)
print('Run directory:', ROOT)
clean, _ = thesis.inputs()
source = thesis.norm(pd.read_csv(ROOT/'inputs'/'consistent_model_data.csv', dtype={'fips': str}))
nov = thesis.norm(pd.read_csv(ROOT/'inputs'/'county_climate_panel_v4.csv', dtype={'fips': str}))
source = source.drop(columns=thesis.TEMP[:2], errors='ignore').merge(
    nov[thesis.KEYS + thesis.TEMP[:2]], on=thesis.KEYS, validate='one_to_one'
)
assert len(clean) == len(source) == 400
assert clean[thesis.KEYS].equals(source[thesis.KEYS]), 'Original and cleaned row order differs'
assert len(thesis.ML) == 21
for column in thesis.ML + list(thesis.TARGETS.values()):
    assert np.array_equal(clean[column].to_numpy(), source[column].to_numpy()), f'Values differ: {column}'
print('PASS: original and cleaned 21 features, targets, and row order are identical')

rows=[]
for threshold, target in thesis.TARGETS.items():
    for mode in ('LOYO', 'Forward'):
        runs={}
        for label, frame in [('original', source), ('cleaned', clean)]:
            pred=np.full(len(frame), np.nan)
            if mode=='LOYO':
                folds=[(frame.year.ne(y).to_numpy(), frame.year.eq(y).to_numpy()) for y in sorted(frame.year.unique())]
            else:
                folds=[(frame.year.le(end).to_numpy(), frame.year.between(start,stop).to_numpy()) for end,start,stop in thesis.FOLDS]
            for tr,te in folds:
                model=thesis.estimator('XGBoost')
                model.fit(frame.loc[tr,thesis.ML], frame.loc[tr,target])
                pred[te]=model.predict(frame.loc[te,thesis.ML])
            runs[label]=pred
        if not np.array_equal(runs['original'], runs['cleaned'], equal_nan=True):
            raise AssertionError(f'Original vs cleaned predictions differ for {threshold} cm {mode}')
        mask=np.isfinite(runs['cleaned'])
        y=clean.loc[mask,target]
        p=runs['cleaned'][mask]
        r={'threshold_cm':threshold,'validation':mode,'R2':r2_score(y,p),
           'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
           'n_obs_scored':int(mask.sum()),'xgboost_version':xgboost.__version__}
        rows.append(r)
        print(f'PASS {threshold}cm {mode}: identical original/cleaned predictions; R2={r["R2"]:.6f}, MAE={r["MAE"]:.6f}, RMSE={r["RMSE"]:.6f}')
        pd.DataFrame({'fips':clean.loc[mask,'fips'],'year':clean.loc[mask,'year'],
                      'threshold_cm':threshold,'validation':mode,'observed':y,'predicted':p}).to_csv(
            OUT/f'xgboost_{threshold}cm_{mode.lower()}_predictions.csv', index=False)

summary=pd.DataFrame(rows)
summary.to_csv(OUT/'xgboost_comparison.csv',index=False)
# Compare any historical robustness results only as reference, not as a fresh run.
old=ROOT/'outputs'/'snodas_remaining_robustness_v4'/'robustness_validation_results.csv'
if old.exists():
    reference=pd.read_csv(old)
    model_col='Model' if 'Model' in reference.columns else 'model'
    for r in rows:
        ref=reference[(reference['threshold_cm']==r['threshold_cm']) &
                      (reference['validation']==r['validation']) &
                      (reference[model_col]=='XGBoost')]
        if len(ref)==1:
            print(f'Historical robustness file {r["threshold_cm"]}cm {r["validation"]}: R2={float(ref.iloc[0]["R2"]):.6f}; fresh R2={r["R2"]:.6f}')
print('Saved diagnostic results to:',OUT)
