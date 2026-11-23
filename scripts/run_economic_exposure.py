"""Economic sensitivity analysis for SNODAS 10 cm and 30 cm warming scenarios.

Requires pandas and numpy. No notebook variables needed.
Run: python run_snodas_economic_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'outputs'
INPUTS = {
    '10cm': BASE / 'snodas_10cm_v4_november' / 'snodas_state_warming_summary.csv',
    '30cm': BASE / 'snodas_30cm_v4_november' / 'snodas_state_warming_summary.csv',
}
OUTPUT = BASE / 'snodas_economic_analysis_v4'
OUTPUT.mkdir(parents=True, exist_ok=True)

# BEA ORSA 2023, skiing/snowboarding value added, millions of current USD.
# Confirmed against the thesis's orsa.csv, LineCode 19 (not LineCode 18).
BEA_VA_M = {'Colorado':1085.550, 'Idaho':90.582, 'Montana':48.993,
            'Utah':419.487, 'Wyoming':27.216}
FIPS_STATE = {8:'Colorado',16:'Idaho',30:'Montana',49:'Utah',56:'Wyoming'}
ELASTICITIES = (0.25, 0.50, 1.00)

assert abs(sum(BEA_VA_M.values()) - 1671.828) < 1e-8
rows = []
for threshold, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(f'Missing {threshold} file: {path}')
    df = pd.read_csv(path)
    required = {'state','warming_C','Model','predicted_days','baseline_days','pct_change'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: missing columns {sorted(required-set(df.columns))}')
    df['state_name'] = df['state'].astype(int).map(FIPS_STATE)
    if df['state_name'].isna().any():
        raise ValueError(f'Unexpected state codes in {path}: {df.loc[df.state_name.isna(),"state"].unique()}')
    if df.duplicated(['state','warming_C','Model']).any():
        raise ValueError(f'Duplicate state/model/scenario rows in {path}')
    if (df['baseline_days'] <= 0).any():
        raise ValueError(f'Nonpositive baseline days in {path}')
    computed_pct = 100 * (df['predicted_days']/df['baseline_days'] - 1)
    if not np.allclose(computed_pct, df['pct_change'], atol=1e-4):
        raise ValueError(f'pct_change inconsistent with predicted_days/baseline_days in {path}')
    expected_states = set(BEA_VA_M)
    if set(df.state_name) != expected_states:
        raise ValueError(f'Expected five states, found {set(df.state_name)}')
    for _, r in df.iterrows():
        for eta in ELASTICITIES:
            baseline = BEA_VA_M[r.state_name]
            change_pct = 100 * (r.predicted_days/r.baseline_days-1)
            delta_va = baseline * eta * change_pct / 100
            rows.append({
                'threshold':threshold, 'state':r.state_name, 'state_fips':int(r.state),
                'Model':r.Model, 'warming_C':int(r.warming_C), 'elasticity':eta,
                'bea_2023_value_added_M':baseline,
                'predicted_days':r.predicted_days, 'baseline_days':r.baseline_days,
                'snow_change_pct':change_pct, 'change_value_added_M':delta_va,
                'implied_value_added_M':baseline+delta_va,
            })

detail = pd.DataFrame(rows)
expected = 2 * 5 * 5 * 4 * len(ELASTICITIES)
if len(detail) != expected:
    raise ValueError(f'Expected {expected} state-level rows, found {len(detail)}')

five_state = (detail.groupby(['threshold','Model','warming_C','elasticity'], as_index=False)
              .agg(change_value_added_M=('change_value_added_M','sum'),
                   implied_value_added_M=('implied_value_added_M','sum'),
                   bea_2023_value_added_M=('bea_2023_value_added_M','sum')))
five_state['change_pct_of_2023_VA'] = (100 * five_state.change_value_added_M /
                                       five_state.bea_2023_value_added_M)

state_path = OUTPUT / 'economic_exposure_by_state.csv'
total_path = OUTPUT / 'economic_exposure_five_state_totals.csv'
detail.to_csv(state_path,index=False)
five_state.to_csv(total_path,index=False)

print('\nBEA ORSA 2023 skiing/snowboarding value added ($M):')
for s, v in BEA_VA_M.items(): print(f'  {s:10s} {v:,.3f}')
print(f'  TOTAL      {sum(BEA_VA_M.values()):,.3f}')

for threshold in INPUTS:
    subset = five_state[(five_state.threshold==threshold) &
                        (five_state.elasticity==0.50) &
                        (five_state.warming_C>0)]
    print(f'\n{threshold} SNODAS: five-state change in value added ($M), elasticity=0.50')
    table = subset.pivot(index='Model',columns='warming_C',values='change_value_added_M')
    print(table.rename(columns=lambda c:f'+{c} C').round(2).to_string())

print(f'\nSaved: {state_path}\nSaved: {total_path}')
print('Negative changes indicate illustrative reductions relative to 2023, not causal damages.')
