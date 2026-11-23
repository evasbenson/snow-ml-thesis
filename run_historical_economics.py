"""Historical economic associations using final SNODAS panel.

Run from project root: python scripts/run_historical_economics.py
Optional GDP files: inputs/gdp/CAGDP2_*.csv (original BEA county exports).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'historical_economic_analysis'
OUT.mkdir(parents=True,exist_ok=True)
SNOW=ROOT/'inputs'/'county_snodas_10cm_30cm_merged.csv'
snow=pd.read_csv(SNOW,dtype={'fips':str})
snow['fips']=snow.fips.str.zfill(5)
assert len(snow)==400 and snow.fips.nunique()==20 and snow.year.nunique()==20

# Rocky Mountain regional skier visits, by season-ending year, transcribed from
# original notebook cell 152; verify against original NSAA source before publication.
visits=[18.868,19.606,20.717,20.849,21.324,19.974,20.378,20.900,
        19.130,19.800,21.100,20.768,22.287,21.736,20.792,24.408,
        20.107,22.638,25.250,28.174,26.757,26.503]
nsaa=pd.DataFrame({'year':range(2004,2026),'skier_visits_millions':visits})
regional=snow.groupby('year',as_index=False).agg(snow_days_10cm=('snow_days_10cm_2000m','mean'))
reg=regional.merge(nsaa,on='year',validate='one_to_one')
reg=reg[(reg.snow_days_10cm>0)&(reg.skier_visits_millions>0)].copy()
reg['log_visits']=np.log(reg.skier_visits_millions)
reg['log_snow']=np.log(reg.snow_days_10cm)
reg['time_trend']=reg.year-reg.year.min()
reg['covid']=reg.year.isin([2020,2021]).astype(int)
reg.to_csv(OUT/'nsaa_snodas_regional_panel.csv',index=False)
rows=[]
for label,formula,exclude in [
 ('Snow only','log_visits ~ log_snow',False),
 ('Snow + trend','log_visits ~ log_snow + time_trend',False),
 ('Snow + trend + COVID','log_visits ~ log_snow + time_trend + covid',False),
 ('Snow + trend, excluding COVID','log_visits ~ log_snow + time_trend',True)]:
 sample=reg.loc[~reg.year.isin([2020,2021])].copy() if exclude else reg
 m=smf.ols(formula,data=sample).fit(cov_type='HC3')
 rows.append({'Specification':label,'N':int(m.nobs),'Elasticity':m.params['log_snow'],
              'Robust SE':m.bse['log_snow'],'P-value':m.pvalues['log_snow'],
              'CI lower':m.conf_int().loc['log_snow',0],'CI upper':m.conf_int().loc['log_snow',1],
              'R2':m.rsquared})
nsaa_results=pd.DataFrame(rows)
nsaa_results.to_csv(OUT/'nsaa_regression_results.csv',index=False)
print('\nNSAA regression results (not causal):\n',nsaa_results.round(4).to_string(index=False))

files=sorted((ROOT/'inputs'/'gdp').glob('CAGDP2_*.csv'))
if not files:
 print('\nGDP NOT RUN: add original CAGDP2_*.csv files to inputs/gdp/.')
else:
 frames=[]
 for path in files:
  x=pd.read_csv(path,dtype={'GeoFIPS':str},encoding='utf-8-sig')
  x['GeoFIPS']=x.GeoFIPS.astype(str).str.replace('"','',regex=False).str.strip().str.zfill(5)
  x=x[x.GeoFIPS.str.fullmatch(r'\d{5}')].copy()
  x['LineCode']=pd.to_numeric(x.LineCode,errors='coerce')
  frames.append(x)
 raw=pd.concat(frames,ignore_index=True)
 codes={1:'total_gdp',75:'tourism_broad_gdp',76:'arts_recreation_gdp',79:'accommodation_food_gdp'}
 years=[str(y) for y in range(2004,2024)]
 missing=[y for y in years if y not in raw]
 if missing:raise ValueError(f'Missing GDP years: {missing}')
 x=raw[raw.LineCode.isin(codes)].copy()
 x['industry']=x.LineCode.map(codes)
 long=x.melt(id_vars=['GeoFIPS','industry'],value_vars=years,var_name='year',value_name='gdp_thousands')
 long['year']=long.year.astype(int)
 long['gdp_thousands']=pd.to_numeric(long.gdp_thousands,errors='coerce')
 # Explicitly reject ambiguous duplicate values rather than silently taking first.
 conflicts=long.groupby(['GeoFIPS','year','industry']).gdp_thousands.nunique(dropna=True)
 if (conflicts>1).any():raise ValueError('Conflicting duplicate GDP values in source exports')
 wide=long.pivot_table(index=['GeoFIPS','year'],columns='industry',values='gdp_thousands',aggfunc='first').reset_index()
 wide.columns.name=None
 panel=snow.merge(wide,left_on=['fips','year'],right_on=['GeoFIPS','year'],how='left',validate='one_to_one')
 panel.to_csv(OUT/'county_snodas_gdp_panel.csv',index=False)
 results=[]
 for var in codes.values():
  if var not in panel:continue
  for label,exclude in [('Full sample',False),('Exclude 2020-2021',True)]:
   sample=panel.loc[(panel[var]>0)&panel.snow_days_10cm_2000m.notna()].copy()
   if exclude:sample=sample.loc[~sample.year.isin([2020,2021])].copy()
   if sample.fips.nunique()<2 or sample.year.nunique()<3:continue
   sample['log_gdp']=np.log(sample[var])
   fit=smf.ols('log_gdp ~ snow_days_10cm_2000m + C(fips) + C(year)',data=sample).fit(cov_type='cluster',cov_kwds={'groups':sample.fips})
   key='snow_days_10cm_2000m'
   results.append({'Outcome':var,'Specification':label,'Observations':int(fit.nobs),'Counties':sample.fips.nunique(),
                   'Coefficient':fit.params[key],'Clustered SE':fit.bse[key],'P-value':fit.pvalues[key],
                   '10-day effect (%)':100*np.expm1(10*fit.params[key])})
 pd.DataFrame(results).to_csv(OUT/'county_gdp_regression_results.csv',index=False)
 print('\nGDP regression results:\n',pd.DataFrame(results).round(5).to_string(index=False))
