# Ski thesis — consolidated analysis

This folder contains a notebook-independent implementation of the final **SNODAS 10 cm and 30 cm** predictive models and the associated warming and economic-exposure analyses. The old notebook and two superseded comparison scripts are preserved in `archive/`.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python run_thesis.py
```

For a faster run that produces the warming and economic outputs without cross-validation:

```bash
python run_thesis.py --skip-validation --skip-robustness
```

The full run may take several minutes or longer, depending on hardware. No original notebook variables or hard-coded Mac paths are needed. Input files are in `inputs/`; results are written to `outputs/`.

## Mapping to thesis

| Thesis section | Implementation |
|---|---|
| 3.1–3.4, 3.6 | Historical county-winter panel: `inputs/county_snodas_10cm_30cm_merged.csv`, November Daymet variables from `county_climate_panel_v4.csv` |
| 4.2 | FE1 (PDD and three precipitation periods), FE2 (+FDD and thaw days), FE3 (+PDD squared); county and year FE with county-clustered standard errors in final fits |
| 4.3 | Random Forest (500 trees, min leaf 3, max features 0.7) and XGBoost (300 estimators, depth 2, learning rate 0.03, subsample 0.8, colsample 0.8, min child weight 5, L2 10), 21 predictors |
| 4.4 | LOYO, three forward splits, LOCO with pooled linear benchmarks, and 392-row coverage sensitivity; validation fits omit year FE |
| 4.5 | Scenario 0/+1/+2/+3°C, November–April seasonal temperatures shifted, daily-derived PDD/FDD/thaw values loaded from original warming CSV, precipitation/terrain unchanged, predictions clipped to `[0, season_days]` |
| 4.6.3 | BEA ORSA 2023 skiing/snowboarding value added, illustrative elasticities 0.25/0.50/1.00, five-state aggregation |

**Scope limitations:** This script does not reproduce the original Earth Engine SNODAS/Daymet extraction, the daily temperature recalculation underlying `county_warming_scenarios_all.csv`, the county GDP regressions (4.6.1), or the NSAA skier-visitation regressions (4.6.2). Those are distinct upstream or economic workflows and should not be represented as covered by this consolidated script. Original economic scripts remain available separately in the user's research project.

## Verification status

- Checked 400 county-winters, 20 counties, 2004–2023.
- Checked 21 ML predictors and 1,600 scenario rows.
- Checked scenario-zero PDD/FDD/thaw match historical values to numerical precision.
- Checked seasonal temperatures increase exactly by the scenario amount, and precipitation and terrain are unchanged.
- Reproduced the historical snow-cover means of 141.1494 days (10 cm) and 116.2371 days (30 cm).
- **Fast scenario/economic run completed.** The FE warming results reproduce previously reported figures to displayed precision. The RF results are very close. XGBoost scenario predictions differ somewhat from previously recorded values; exact reproduction of the older ML outputs remains **unverified**, potentially due to library versions or other execution-environment differences. Do not silently overwrite prior thesis figures without comparing the old and new CSV outputs.
- **Full validation and robustness run not completed in this environment** (execution timed out). The code is provided for running locally; those metrics should be checked against existing results after completion.

## Note on model interpretation

The FE validation models use county indicators without year indicators, while full-sample warming fits use both county and year indicators. LOCO uses pooled linear models with no county indicators. The economic elasticities are **assumptions**, not estimated from the NSAA or county GDP regressions. Warming scenarios are sensitivity exercises, not forecasts or causal damages.

## Manuscript corrections recommended

- Section 4.5's change-in-snow-cover equation should be `ΔŜ_it = Ŝ_it^(Δ) − Ŝ_it^(0)`; the pasted equation currently lost its minus sign and some formatting.
- Section 4.6.3's snow change formula should be `g_s^(Δ) = (mean Ŝ_s^(Δ) / mean Ŝ_s^(0)) − 1` and the value-added change formula should be `ΔVA_s^(Δ) = VA_(s,2023) × η × g_s^(Δ)`.
- Section 4.6 is missing its parent heading in the pasted manuscript. Add `### 4.6 Economic Analysis` before the introductory paragraph.
- Section 4.5 has a grammatical issue: change “This allows for the temperature-dependent predictors remain consistent” to “This keeps the temperature-dependent predictors consistent.”
- Section 3.3 says “daily minimum temperature, maximum temperature, and precipitation were extracted for each county and averaged across areas at or above 2,000 meters.” That is consistent with the input structure; the original Earth Engine source is needed to audit the upstream aggregation independently.

## November 21 update — XGBoost row-order test

Removed unnecessary sorting of training rows from `run_thesis.py` and both standalone SNODAS scripts to test whether input row ordering explains the small XGBoost differences. This is a reproducibility hypothesis, **not yet a verified correction**. Rerun `python3 run_thesis.py` and compare the newly generated XGBoost validation and warming values with the original results. All model hyperparameters and data are unchanged.
