# Thesis-ready figures and tables, run AFTER the model and warming notebook cells.
# Required: model_df, snow_predictions, state_pct_summary, economic_results,
#           economic_totals, forward_matched, spatial_predictions.
# Optional: orsa_2023, county shapefile (see map section).
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUT = Path("thesis_outputs")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["Fixed Effects", "Random Forest", "XGBoost"]
PRED_COLS = {"Fixed Effects": "fe_predicted",
             "Random Forest": "rf_predicted",
             "XGBoost": "xgb_predicted"}
TARGET = "snow_days_10cm_2000m"
STATE_FIPS = {"08":"Colorado", "16":"Idaho", "30":"Montana",
              "49":"Utah", "56":"Wyoming"}

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.show()

def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n{name}\n", df.round(3).to_string(index=False))

def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    return dict(N=len(a), R2=r2_score(a,p),
                MAE=mean_absolute_error(a,p),
                RMSE=np.sqrt(mean_squared_error(a,p)))

def check(names):
    missing = [n for n in names if n not in globals()]
    if missing:
        raise RuntimeError(f"Run earlier notebook cells first. Missing: {missing}")

check(["model_df", "snow_predictions", "state_pct_summary",
       "economic_results", "economic_totals",
       "forward_matched", "spatial_predictions"])

df = model_df.copy()
df["fips"] = df["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
df["year"] = pd.to_numeric(df["year"]).astype(int)
df["state_name"] = df["fips"].str[:2].map(STATE_FIPS)
assert not df.duplicated(["fips","year"]).any(), "Duplicate county-years"
assert all(c in df.columns for c in [TARGET, *PRED_COLS.values()])
assert len(df) == 400, f"Expected 400 observations; found {len(df)}"

# TABLE 1: Descriptive statistics. Edit columns to suit manuscript.
descriptive_cols = [TARGET, "valid_fraction_2000m",
    "positive_degree_days_2000m", "freezing_degree_days_2000m",
    "thaw_days_2000m", "precip_djf_2000m", "precip_mar_apr_2000m",
    "mean_elevation_m", "mean_slope_deg"]
descriptive_cols = [c for c in descriptive_cols if c in df.columns]
desc = df[descriptive_cols].describe().T.reset_index().rename(
    columns={"index":"Variable", "count":"N", "mean":"Mean", "std":"SD",
             "min":"Min", "max":"Max"})
save_table(desc[["Variable","N","Mean","SD","Min","Max"]],
           "table_1_descriptive_statistics")

# FIGURE 2: Historical snow-cover days, annual mean across sample counties.
annual = df.groupby("year", as_index=False)[TARGET].mean()
save_table(annual, "data_figure_2_annual_snow")
fig, ax = plt.subplots(figsize=(8,4.6))
ax.plot(annual["year"], annual[TARGET], marker="o")
ax.set(xlabel="Winter ending year", ylabel="Mean snow-cover days",
       title="Historical high-elevation snow-cover duration")
ax.grid(alpha=.25)
savefig("figure_2_historical_snow")

# FIGURE 3: LOYO observed versus predicted (out-of-fold only).
fig, axes = plt.subplots(1,3,figsize=(13,4.1),sharex=True,sharey=True)
bounds = [0, max(185, df[TARGET].max(),
                 *(df[c].max() for c in PRED_COLS.values()))]
for ax,(name,col) in zip(axes,PRED_COLS.items()):
    ax.scatter(df[TARGET], df[col], s=14, alpha=.5)
    ax.plot(bounds,bounds,"k--",lw=1)
    m=metrics(df[TARGET],df[col])
    ax.set(title=f"{name}\nR²={m['R2']:.3f}; MAE={m['MAE']:.2f}",
           xlabel="Observed snow-cover days", xlim=bounds,ylim=bounds)
    ax.grid(alpha=.15)
axes[0].set_ylabel("LOYO predicted snow-cover days")
savefig("figure_3_observed_vs_predicted")

# TABLE 2: Three validation methods, noting linear-model differences.
records=[]
for name,col in PRED_COLS.items():
    records.append({"Validation":"Leave-one-year-out", "Model":name,
                    **metrics(df[TARGET],df[col])})
for name,preds in forward_matched.items():
    records.append({"Validation":"Forward-in-time", "Model":name,
                    **metrics(df[TARGET],preds)})
for name,preds in spatial_predictions.items():
    records.append({"Validation":"Leave-one-county-out", "Model":name,
                    **metrics(df[TARGET],preds)})
validation = pd.DataFrame(records)
save_table(validation, "table_2_model_validation")

# APPENDIX: Box Elder sensitivity using existing spatial predictions.
mask = df["county"].astype(str).str.casefold().ne("box elder").to_numpy()
box = pd.DataFrame([
    {"Model":name, **metrics(df.loc[mask,TARGET],
                            np.asarray(preds)[mask])}
    for name,preds in spatial_predictions.items()])
save_table(box, "appendix_box_elder_sensitivity")

# APPENDIX FIGURE: Validation RMSE, RF and XGB only, differing test samples.
comparison = validation[validation["Model"].isin(["Random Forest","XGBoost"])]
pivot = comparison.pivot(index="Validation",columns="Model",values="RMSE")
pivot = pivot.reindex(["Leave-one-year-out","Forward-in-time",
                       "Leave-one-county-out"])
pivot.plot(kind="bar",figsize=(8,4.6),rot=0)
plt.ylabel("RMSE (snow-cover days)")
plt.title("Out-of-sample error across validation designs")
plt.grid(axis="y",alpha=.2)
savefig("appendix_validation_rmse")

# APPENDIX FIGURE: County-level LOYO MAE, three models.
county_rows=[]
for (fips,county),g in df.groupby(["fips","county"],dropna=False):
    for name,col in PRED_COLS.items():
        county_rows.append({"fips":fips,"County":county,"Model":name,
                            "MAE":mean_absolute_error(g[TARGET],g[col])})
county_errors=pd.DataFrame(county_rows)
save_table(county_errors,"appendix_county_prediction_errors")
# County names can repeat across states; FIPS is the unique county identifier.
county_errors["County_Label"] = (
    county_errors["County"].astype(str) + " (" +
    county_errors["fips"].astype(str).str.zfill(5) + ")"
)
county_errors.pivot(index="County_Label", columns="Model", values="MAE").sort_values(
    "Random Forest").plot(kind="barh", figsize=(9,8))
plt.xlabel("LOYO mean absolute error (days)")
plt.title("Prediction error by county")
plt.grid(axis="x",alpha=.2)
savefig("appendix_county_mae")

# APPENDIX FIGURE: residual patterns, model prediction minus observed.
fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True)
for ax,(name,col) in zip(axes,PRED_COLS.items()):
    residual=df[col]-df[TARGET]
    ax.scatter(df[TARGET],residual,s=14,alpha=.45)
    ax.axhline(0,color="black",ls="--",lw=1)
    ax.set(title=name,xlabel="Observed snow-cover days",
           ylabel="Prediction error (days)")
    ax.grid(alpha=.2)
savefig("appendix_residuals")

# TABLE 3 + FIGURE 4: Mean snow-day change under warming.
snow = snow_predictions.copy()
assert set([0,1,2,3]).issubset(set(snow["warming_C"]))
change_cols=[f"{m}_change" for m in MODELS]
warming = snow.groupby("warming_C",as_index=False)[change_cols].mean()
warming = warming.rename(columns={f"{m}_change":m for m in MODELS})
save_table(warming,"table_3_warming_snow_days")
fig,ax=plt.subplots(figsize=(8,4.6))
for model in MODELS:
    ax.plot(warming["warming_C"],warming[model],marker="o",label=model)
ax.axhline(0,color="black",lw=.7)
ax.set(xlabel="Uniform warming (°C)",
       ylabel="Change in mean snow-cover days",
       title="Predicted response to uniform warming")
ax.set_xticks([0,1,2,3])
ax.legend()
ax.grid(alpha=.2)
savefig("figure_4_warming_response")

# FIGURE 5: State proportional snow changes at +3 C.
state_pct = state_pct_summary.copy()
at3 = state_pct.loc[state_pct["warming_C"].eq(3),
                   ["state",*MODELS]].set_index("state")
save_table(at3.reset_index(),"data_figure_5_state_pct_at_3C")
at3.reindex(["Colorado","Idaho","Montana","Utah","Wyoming"]).plot(
    kind="bar",figsize=(9,4.7),rot=0)
plt.ylabel("Change in snow-cover duration (%)")
plt.title("State sample response to +3°C warming")
plt.grid(axis="y",alpha=.2)
savefig("figure_5_state_warming")

# COUNTY-LEVEL WARMING RESULTS: aggregate by county and model first,
# then divide summed changes by summed model baselines.
county_snow = snow.copy()
county_snow["fips"] = county_snow["fips"].astype(str).str.zfill(5)
county_records=[]
for (fips,county,w),g in county_snow.groupby(
        ["fips","county","warming_C"],dropna=False):
    if w == 0: continue
    for model in MODELS:
        b=g[f"{model}_baseline"].sum()
        change=g[f"{model}_change"].sum()
        county_records.append({"fips":fips,"County":county,"Warming_C":w,
            "Model":model,"Mean_change_days":g[f"{model}_change"].mean(),
            "Pct_change":100*change/b if b>0 else np.nan})
save_table(pd.DataFrame(county_records),"appendix_county_warming")

# TABLE 4: Economic totals, all elasticities, all models.
economics = economic_totals.copy().sort_values(
    ["warming_C","Model","elasticity"])
save_table(economics,"table_4_economic_sensitivity_all_elasticities")
central = economics.loc[np.isclose(economics["elasticity"],.50)].copy()
save_table(central,"table_4b_economic_sensitivity_central")

# 2023 BEA baseline if available.
if "orsa_2023" in globals():
    base = orsa_2023[["state","baseline_va_millions"]].copy()
    save_table(base,"table_economic_2023_baseline")

# Optional FIGURE 1: County map.
# Requires a local U.S. county boundaries shapefile/GeoPackage.
# Supply your own path, e.g. Path("tl_2023_us_county.shp").
COUNTY_BOUNDARIES = None
if COUNTY_BOUNDARIES is not None:
    import geopandas as gpd
    counties = gpd.read_file(COUNTY_BOUNDARIES)
    if "GEOID" not in counties:
        raise ValueError("County boundaries need a GEOID column")
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    study = counties.loc[counties["GEOID"].isin(df["fips"])].copy()
    assert study["GEOID"].nunique()==df["fips"].nunique(), (
        "Boundary file does not contain all study counties")
    background = counties.loc[counties["GEOID"].str[:2].isin(
        STATE_FIPS)].copy()
    ax=background.plot(color="#eeeeee",edgecolor="#bbbbbb",
                       linewidth=.35,figsize=(8,7))
    study.plot(ax=ax,color="#4477aa",edgecolor="#222222",linewidth=.6)
    ax.set_axis_off()
    ax.set_title("Study counties in the western United States")
    savefig("figure_1_study_area")
else:
    print("\nMap skipped: set COUNTY_BOUNDARIES to a local county shapefile.")

print(f"\nFinished. Figures (PNG/PDF) and tables (CSV): {OUT.resolve()}")
