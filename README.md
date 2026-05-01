# A Geospatial Equity Analysis of Boston Bus Performance
### CS 506 Final Report

**Team:** Baria Mustafa · Primah Muwanga · Amira Zuniga  
**GitHub:** https://github.com/azuniga1117/CS506-Final-Project

**Google Drive:** https://drive.google.com/drive/folders/1YNus7FoPNNob80MJNP2GP7s-mLP01nul?usp=sharing

---

## Table of Contents
1. [How to Build and Run the Code](#how-to-build-and-run-the-code)
2. [Project Description](#project-description)
3. [Project Goals](#project-goals)
4. [Data Collection](#data-collection)
5. [Data Cleaning](#data-cleaning)
6. [Feature Extraction](#feature-extraction)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Cross-Model Comparison](#cross-model-comparison)
9. [Data Visualization & Results](#data-visualization--results)
10. [Team](#team)

---

## How to Build and Run the Code

### Requirements

- Python 3.10+
- Google Colab (recommended) with Drive mounted, or a local environment with `mbta_final.csv`

### Setup & Run

**XGBoost** *(Baria Mustafa)*
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn shap
```
Open `XGBoost-5.ipynb` in Google Colab and run all cells top-to-bottom.

**Linear Regression** *(Primah Muwanga)*
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```
Open the linear regression notebook in Google Colab and run all cells top-to-bottom.

**Random Forest** *(Amira Zuniga)*
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
Open the Random Forest notebook in Google Colab and run all cells top-to-bottom.

### Reproducing Results

All results are reproducible with `RANDOM_STATE = 42`. The XGBoost pipeline uses a stratified random sample of 500,000 rows from the full ~26M-row dataset (stratified by `route_id`), so results are stable across runs.

---

## Project Description

This project investigates service disparities within the MBTA bus network, examining how geography and socioeconomic demographics correlate with transit reliability across 2023–2024 post-pandemic data. Despite the MBTA serving over one million riders daily and contributing an estimated $11.5 billion in annual economic value, service quality is distributed unevenly. Prior research — including the 2017 *"64 Hours"* report — found that Black bus riders spent 64 more hours per year commuting than white riders due to systemic service gaps.

This project updates that analysis using three complementary machine learning approaches — Linear Regression, Random Forest, and XGBoost — to predict average bus delay at the route-stop-hour level, with equity-focused predictors including Title VI low-income and minority ridership percentages from the MBTA Passenger Survey. The goal is to move beyond descriptive statistics toward predictive models that quantify which operational and demographic factors drive the "time tax" on transit-dependent communities.

---

## Project Goals

| Scope | Goal |
|---|---|
| **Team (Primary)** | Quantify the reliability gap by calculating average delay time and wait-time variance for the 15 target bus routes (28, 23, 111, 15, and others), identifying which routes fall more than 20% below MBTA service standards |
| **Team (Secondary)** | Correlate bus performance metrics (delays, travel speed, frequency) with neighborhood demographics (race, income, vehicle ownership) to determine if a statistically significant disparity persists in the time tax paid by low-income and minority residents |
| **XGBoost (Baria)** | Predict `avg_delay_min` at the route × stop × hour × day-of-week level with R² > 0.5 on held-out data; determine via forward adjusted-R² selection whether Title VI demographic features improve model fit after controlling for operational factors |
| **Linear Regression (Primah)** | Establish whether demographic factors (Title VI minority %, low-income %) and operating factors (hour, rush hour, previous delay, route) significantly explain trip-level delay via OLS; quantify how much variance the demographic group adds using a partial F-test (α = 0.05) |
| **Random Forest (Amira)** | Capture non-linear interactions between route characteristics, time-of-day, and demographics using a Random Forest Regressor on route-level aggregated data, targeting R² > 0.40; quantify whether demographic features contribute meaningfully to feature importance relative to operational predictors |

---

## Data Collection

### Data Sources

| Source | Description | How Accessed |
|---|---|---|
| MBTA Bus Arrival/Departure Times (2018–2024) | Core performance dataset (~26M rows): scheduled and actual timestamps, `route_id`, `stop_id`, direction, neighborhood, municipality | Bulk CSV — MassGIS / MBTA Open Data Portal |
| MBTA 2024 System-Wide Passenger Survey | Rider demographics by bus reporting group: Title VI Low-Income % and Title VI Minority % at the route level (`weighted_percent`) | CSV — MBTA Open Data Portal |
| MBTA Bus Trip-Level CSV (subset) | 325,924 trip observations with `service_date`, `route_id`, `direction_id`, `neighborhood`, `hour`, `day_of_week`, `is_rush_hour`, `prev_delay`, `pct_minority`, `pct_lowincome`, `delay_seconds` | Bulk CSV — same MBTA Open Data source |
| MBTA Route-Level Aggregated Dataset | 226 route-level observations with `avg_delay` as response; includes engineered features `route_historical_avg_delay`, `headway_deviation`, `stop_position_pct` | Derived from MBTA bulk CSV — aggregated to route level |

### Collection Method

**XGBoost** *(Baria Mustafa)*  
The MBTA arrival/departure dataset was downloaded as a bulk CSV. Due to its ~26M-row scale, a proportional stratified random sample of 500,000 rows was drawn at runtime using a chunked streaming reader (1M rows/chunk), stratified by `route_id` to preserve each route's share of the total. The Passenger Survey CSV was filtered to Bus service mode, Reporting Group aggregation level, and Title VI Low-Income / Title VI Minority measures. Grouped reporting entries (e.g., "28 & 29") were expanded into individual route rows via regex splitting.  
*See `XGBoost-5.ipynb` §1 (stratified sampler) and §2a (demographic loader).*

**Linear Regression** *(Primah Muwanga)*  
TThe trip-level CSV `trip_level_merged.csv` was loaded with `pd.read_csv()`, producing 325,924 rows across 11 columns. No additional sampling was applied at this stage, the upstream 800,000-row uniform random sample of `mbta_final.csv` (drawn in `merge_for_regression.py` with `np.random.choice` and a fixed seed of 42) yields the trip-level dataset directly. Demographic columns (`pct_minority`, `pct_lowincome`) were pre-merged at the route level from the Passenger Survey, with grouped survey labels (e.g., `'114, 116, & 117'`) regex-expanded into individual route rows before the join. See `merge_for_regression.py` for the upstream pipeline and `lin_reg_final.py` for the regression load step.

**Random Forest** *(Amira Zuniga)*  
The dataset was aggregated to the route level from the shared MBTA trip CSV, producing 226 observations. Route-level aggregation was chosen to match the granularity of the demographic features, which are only available at the route level. Engineered features (`route_historical_avg_delay`, `headway_deviation`, `stop_position_pct`) were computed during preprocessing.  
*See Random Forest notebook.*

---

## Data Cleaning

### XGBoost *(Baria Mustafa)*

1. **Timestamp parsing** — `scheduled`, `actual`, and `service_date` parsed with `pd.to_datetime(errors='coerce')` to coerce unparseable values to `NaT`.
2. **Delay clipping** — `delay_seconds` clipped to [−3600, 10800] s (−1 hr to +3 hrs) to remove physically implausible values from data artifacts or cancellations.
3. **Categorical normalization** — string columns lowercased and stripped before label encoding; missing values replaced with `'__missing__'` sentinel.
4. **Demographic expansion** — Passenger Survey reporting groups (e.g., "28 & 29") split on commas/ampersands so each route receives the group's `weighted_percent`.
5. **Aggregation** — trip-level observations aggregated to route × stop × hour × day-of-week level to produce `avg_delay_min`. Reduces noise and matches the grain of the equity analysis.

*Clipping over dropping preserves row count while preventing extreme outliers from distorting gradient boosting splits. See `XGBoost-5.ipynb` §2b–§2c.*

### Linear Regression *(Primah Muwanga)*

1. **NaN removal** — `dropna(subset=[response] + predictors + ["route_id"])` applied; all 325,924 rows retained, confirming no missing values in the pre-merged file. (Missingness was already handled upstream in `merge_for_regression.py`, which dropped first-trip-per-route rows lacking a `prev_delay` value, and at the route level for any route missing demographics.)
2. **Categorical encoding** — `route_id` one-hot encoded via `pd.get_dummies(..., drop_first=True)`, producing 35 binary dummy columns (design matrix: n = 325,924, p = 41 = 6 numeric + 35 dummies). One-hot was chosen over label encoding because OLS interprets numeric predictor values as continuous; label codes (e.g., route 28 = 28, route 39 = 39) would impose an arbitrary linear relationship across routes that doesn't correspond to anything real. `neighborhood` was excluded as a predictor entirely due to a 47.4% Unknown rate in the source data; `direction_id` was used only as a grouping variable for the `prev_delay` lag construction, not as a predictor.
3. **Multicollinearity check** — Pearson correlation matrix computed on all numeric predictors. `pct_minority` and `pct_lowincome` showed r = 0.815. Both were retained in the full model because they capture distinct policy dimensions (race vs. income), but their joint coefficients are unstable and were not interpreted individually; instead, single-predictor refits with operating controls were used to recover stable coefficient estimates (Visualization 5).
4. **Response variable** — `delay_seconds` clipped to ±3600 seconds upstream in `merge_for_regression.py`, removing implausible values from data entry errors and cancelled trips. The coefficient unit (seconds of delay per unit of predictor) remains directly interpretable.

*See `lin_reg_final.py` for design matrix construction and `merge_for_regression.py` for upstream cleaning.*

### Random Forest *(Amira Zuniga)*

1. **Route-level aggregation** — trip-level records aggregated to 226 route-level rows; `avg_delay` computed as mean delay per route, matching Passenger Survey demographic grain.
2. **Categorical label encoding** — `route_id` and `neighborhood` label-encoded (`route_encoded`, `neighborhood_encoded`). Label encoding acceptable for Random Forest (order-invariant splits) and avoids dimensionality explosion at n=226.
3. **Feature engineering** — three features engineered: `route_historical_avg_delay` (per-route mean delay), `headway_deviation` (inter-arrival time variance), `stop_position_pct` (stop index ÷ route length).
4. **Multicollinearity check** — `pct_minority` and `pct_low_income` showed r = 0.719; flagged but retained; RF importance scores are less sensitive to collinearity than OLS coefficients.

*See Random Forest notebook for implementation.*

---

## Feature Extraction

### XGBoost *(Baria Mustafa)*

| Feature | Description | Justification |
|---|---|---|
| `hour` | Hour of day (0–23) | Intra-day congestion patterns; peak hours drive most delay variance |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Weekday vs. weekend service frequency and ridership differ substantially |
| `month` | Calendar month (1–12) | Seasonal variation in ridership, road conditions, construction |
| `is_weekend` | Binary: 1 if Sat/Sun | Strong signal for reduced service frequency |
| `is_peak` | Binary: 1 if hour ∈ [7–9] or [16–18] | MBTA AM/PM rush hours — primary delay amplifier |
| `route_id_str_enc` | Label-encoded route ID | Route-specific delay patterns (frequency, length, traffic exposure) |
| `direction_id_enc` | Label-encoded direction | Inbound/outbound asymmetric congestion |
| `point_type_enc` | Label-encoded stop type | Timepoints are scheduled anchors; non-timepoints accumulate slack differently |
| `standard_type_enc` | Label-encoded service standard | MBTA service classification (Key Route, etc.) |
| `neighborhood_enc` | Label-encoded neighborhood | Localized traffic, road quality, stop spacing |
| `municipality_enc` | Label-encoded municipality | Broader geographic context |
| `time_point_order` | Stop sequence position on route | Later stops accumulate more delay |
| `stop_id_enc` | Label-encoded stop ID | Fine-grained stop-level effects |
| `pct_low_income` | Title VI Low-Income rider share (0–1) | **Equity predictor**: tests if high low-income routes have worse delays |
| `pct_minority` | Title VI Minority rider share (0–1) | **Equity predictor**: tests for racial disparity in reliability |
| `hist_delay_route_hour` | Mean delay for route × hour | Historical baseline — strongest single predictor |
| `hist_delay_route_dow` | Mean delay for route × day-of-week | Day-of-week chronic delay pattern per route |
| `hist_delay_stop` | Mean delay for stop across all observations | Stop-level chronic delay (chronically congested intersections) |

Historical delay features are computed as `transform('mean')` over the full sample — a form of target encoding. This introduces mild data leakage; the adjusted R² forward-selection check (Section 3b of the notebook) was designed to detect whether this inflates apparent performance.

*See `XGBoost-5.ipynb` §2b–§2d.*

### Linear Regression *(Primah Muwanga)*

| Feature | Description | Justification |
|---|---|---|
| `pct_minority` | Title VI Minority rider share (0–1) | **Equity predictor**; r = −0.113 with `delay_seconds` |
| `pct_lowincome` | Title VI Low-Income rider share (0–1) | **Equity predictor**; r = −0.131 with `delay_seconds`; correlated with minority (r = 0.815) |
| `hour` | Hour of departure (0–23) | Temporal congestion signal; r = 0.066 with delay |
| `day_of_week` | Day of week (0–6) | Weekday/weekend service pattern control |
| `is_rush_hour` | Binary: 1 if AM or PM peak | Peak-period delay amplification; coefficient = +39.5 sec |
| `prev_delay` | Delay of preceding trip on same route+direction (sec) | Strongest numeric predictor — cascading delay signal; r = 0.206 |
| `route_id` (dummies) | One-hot encoded route ID (35 dummies after `drop_first=True`) | Route fixed effects — absorbs chronic route-level delay |

The six numeric predictors entered the model without scaling — OLS coefficients are interpretable in original units (seconds per unit change). Design matrix: n = 325,924, p = 41 (6 numeric + 35 dummies). No interaction terms added; intentionally kept as a linear baseline.

`neighborhood` was excluded as a predictor due to a 47.4% Unknown rate in the source data; `direction_id` was used only as a grouping variable for the `prev_delay` lag construction, not as a predictor.

### Random Forest *(Amira Zuniga)*

| Feature | Importance Score | Description |
|---|---|---|
| `route_historical_avg_delay` | **0.4336** | Mean historical delay per route (engineered) |
| `headway_deviation` | 0.1095 | Inter-arrival time variance per route (engineered) |
| `stop_position_pct` | 0.1077 | Relative stop position along route, 0–1 (engineered) |
| `is_rush` | 0.0866 | Binary: 1 if AM or PM peak |
| `hour` | 0.0831 | Hour of departure (0–23) |
| `route_encoded` | 0.0733 | Label-encoded route ID |
| `day_of_week` | 0.0514 | Day of week (0–6) |
| `neighborhood_encoded` | 0.0389 | Label-encoded neighborhood |
| `pct_low_income` | 0.0096 | Title VI Low-Income rider share |
| `pct_minority` | 0.0063 | Title VI Minority rider share |

---

## Model Training & Evaluation

### XGBoost *(Baria Mustafa)*

**Model:** XGBoost Regressor — handles mixed feature types natively, robust to skewed delay distributions, produces SHAP-compatible feature importances, and supports direct regularization to control overfitting.

**Training:** 80/20 train-test split (`random_state=42`) on aggregated `df_agg`. Hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 400 | Convergence on 500K-sample dataset |
| `learning_rate` | 0.08 | Conservative step size to reduce variance |
| `max_depth` | 6 | Up to 6-level interactions without extreme overfit |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `min_child_weight` | 5 | Prevents splits on very small strata |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |

**Evaluation:** MAE, RMSE, R² on held-out 20% test set. Additionally, a forward adjusted-R² selection was run using 5-fold CV to test each feature group's genuine contribution to generalization (CV Test R², Train R², Adjusted R², Train-Test Gap — gap > 0.05 flags overfitting).

**Results:**

| Model | Metric | Score |
|---|---|---|
| XGBoost (`avg_delay_min`) | MAE | **1.796 min** |
| XGBoost (`avg_delay_min`) | RMSE | **2.750 min** |
| XGBoost (`avg_delay_min`) | R² | **0.616** |
| XGBoost (`avg_delay_min`) | Adj R² (CV, best group) | 0.6110 (k=17, historical delay features) |
| Naive baseline | MAE | 4.436 min (std of `avg_delay_min`) |

**Limitations:**
**Adjusted R² Forward-Selection Results (5-fold CV, n=45,264):**

| Feature Group | k | CV Test R² | Train R² | Adj R² | Gap | Verdict |
|---|---|---|---|---|---|---|
| Temporal | 5 | 0.1804 | 0.1880 | 0.1803 | 0.008 | ✅ Useful |
| + Route / Stop / Network | 12 | 0.5680 | 0.6569 | 0.5679 | 0.089 | ✅ Useful ⚠️ overfit gap |
| + Stop ID | 13 | 0.5653 | 0.6672 | 0.5652 | 0.102 | ❌ Hurts — adj R² dropped |
| + Demographic (pct_low_income, pct_minority) | 15 | 0.5814 | 0.6820 | 0.5812 | 0.101 | ✅ Useful ⚠️ overfit gap |
| + Historical delay (route×hour, route×dow) | 17 | 0.6112 | 0.7342 | 0.6110 | 0.123 | ✅ Useful ⚠️ overfit gap |
| + Historical delay (stop) | 18 | 0.6088 | 0.7388 | 0.6086 | 0.130 | ❌ Hurts — adj R² dropped |

**Key findings from the overfitting check:**
- `stop_id_enc` (k=13) **hurts** the model — adj R² drops by 0.003 and the train-test gap exceeds 0.10. Dropped from the recommended feature set.
- Demographic features (`pct_low_income`, `pct_minority`) **meaningfully improve** adj R² (+0.016) even after controlling for all route/network features — confirming they carry signal beyond route structure.
- `hist_delay_stop` (k=18) also **hurts** — adj R² drops 0.002 with a gap of 0.130, consistent with within-sample leakage from computing stop means over the full dataset.
- All groups beyond Temporal show train-test gaps > 0.05, driven primarily by the route/network fixed effects. The recommended final feature set is **k=17** (excluding `stop_id_enc` and `hist_delay_stop`).

- `hist_delay_route_hour` and `hist_delay_stop` are computed over the full sample (not within CV folds), introducing mild data leakage — the adjusted R² check quantifies the impact
- Demographic coverage limited to routes matched in the Passenger Survey
- Route × stop × hour aggregation smooths within-stratum variance; model cannot predict individual trip-level extremes
- No weather, incident, or construction data — primary sources of residual unexplained variance

*See `XGBoost-5.ipynb` §3 (training), §3b (adjusted R² check), §4 (visualizations).*

---

### Linear Regression *(Primah Muwanga)*

**Model:** OLS Linear Regression — provides interpretable coefficient estimates with direct statistical significance testing. The F-test structure allows formal testing of whether demographics alone explain delay, and by how much operating + route factors improve fit.

**Training:** Two OLS models fit on the full 325,924-row dataset using `numpy.linalg.lstsq` to solve the normal equations (more numerically stable than `(XᵀX)⁻¹Xᵀy` when the design matrix is wide or near-singular). No train/test split — the focus was coefficient estimation and hypothesis testing rather than prediction. Demographics-only model (p = 2) fit first, then full model (p = 41). The overall F-statistic tests H₀: all slope coefficients = 0; a separate partial F-test (in `regression_viz.py`) tests the secondary-goal hypothesis that demographics add no information beyond operating factors.

**Evaluation:** R², adjusted R², RMSE (in seconds), F-statistic with p-value. The demographics-only model serves as an equity-focused baseline; the R² gain from operating + route fixed effects measures how much trip-to-trip variance is explained by factors outside the demographic question.

**Results:**

| Model | Metric | Score |
|---|---|---|
| Demographics only (p = 2) | R² | 0.0172 |
| Demographics only (p = 2) | Adj R² | 0.0172 |
| Demographics only (p = 2) | RMSE | 369.18 sec |
| Full model (p = 41) | R² | 0.0893 |
| Full model (p = 41) | Adj R² | 0.0891 |
| Full model (p = 41) | RMSE | 355.42 sec |
| Full model (p = 41) | F-stat (p-value) | 779.0 (p < 0.001) |
| R² gain: operating + route over demographics alone | Δ R² | **+7.20 pp** |

**Key coefficients (full model):**

| Predictor | Coefficient |
|---|---|
| `intercept` | +198.12 sec |
| `pct_minority` | −153.98 sec |
| `pct_lowincome` | +50.66 sec |
| `hour` | +3.62 sec |
| `day_of_week` | +2.76 sec |
| `is_rush_hour` | +39.49 sec |
| `prev_delay` | +0.15 sec |

**Limitations:**

- Full model explains only 8.9% of trip-level delay variance — weather, incidents, and real-time traffic are unobserved
- Multicollinearity between `pct_minority` and `pct_lowincome` (r = 0.815) makes the joint coefficients unstable. From the alone-vs-joint comparison (Visualization 5): `pct_minority` shrinks from −237.99 alone to −37.09 jointly, and `pct_lowincome` shrinks from −504.14 alone to −449.45 jointly. Neither demographic coefficient should be interpreted in isolation from this comparison
- The residual distribution is heavy-tailed (closer to Laplace than Normal) rather than homoscedastic, so standard errors are approximate and the model fits typical trips well but extreme delays poorly (visible in the predicted-vs-actual plot)
- No interaction terms; model cannot detect whether rush-hour penalties or previous-delay propagation differ by demographic group

*See `lin_reg_final.py` for full implementation and output.*

---

### Random Forest *(Amira Zuniga)*

**Model:** Random Forest Regressor (sklearn) — captures non-linear interactions between route characteristics, time-of-day, and demographics without feature scaling. Bagging reduces variance. Produces Gini-based feature importances comparable to XGBoost's gain-based importances.

**Training:** 226-row route-level dataset (n=226, p=10), 80/20 train-test split. Feature importances extracted from `model.feature_importances_` post-fit. F-statistic computed on held-out predictions to test overall significance.

**Evaluation:** R², adjusted R², RMSE, MAE on held-out test set (n≈45). F-statistic tests overall model significance.

**Results:**

| Model | Metric | Score |
|---|---|---|
| Random Forest (p=10, n=226) | R² | 0.4604 |
| Random Forest (p=10, n=226) | Adj R² | 0.4353 |
| Random Forest (p=10, n=226) | RMSE | 107.37 sec |
| Random Forest (p=10, n=226) | MAE | 77.47 sec |
| Random Forest (p=10, n=226) | F-stat (p-value) | 18.34 (p < 0.001) |
| Top feature: `route_historical_avg_delay` | Importance | 0.4336 (43.4%) |

**Limitations:**
- n=226 is small for a Random Forest; the 80/20 split yields ~45 test rows, making metrics sensitive to which routes land in the test set
- `route_historical_avg_delay` dominates at 43.4% importance — the model largely predicts that historically-late routes stay late, which is useful for planning but limits operational actionability
- Demographic features (`pct_minority`: 0.63%, `pct_low_income`: 0.96%) have near-zero importance at the route-aggregated level — this likely reflects absorption by route fixed effects, not absence of disparity
- Multicollinearity between `pct_minority` and `pct_low_income` (r = 0.719) further suppresses their individual importance scores

*See Random Forest notebook for full implementation and output.*

---

## Cross-Model Comparison

| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| Naive Baseline | 254.29 sec | 372.40 sec | 0.000 | Always predicts mean delay (204.16 sec) | 
| Linear Regression (full, p=41) | 241.61 sec | 355.42 sec | 0.0893 (adj: 0.0891) | Trip-level; OLS; interpretable coefficients; F-test; multicollinearity flagged |
| Random Forest (p=10, n=226) | 77.47 sec | 107.37 sec | 0.4604 (adj: 0.4353) | Route-level; highest R²; `route_historical_avg_delay` dominates at 43.4%; demographic features < 1% importance |
| XGBoost | 107.76 sec | 165 sec | 0.616 (adj: 0.6110) | Route-stop-hour level; SHAP interpretability; forward adjusted-R² overfitting check |

> ⚠️ **Note on comparability:** Models are at different aggregation levels — Linear Regression at the trip level (n = 325,924 individual arrivals), Random Forest at the route level (n = 226 routes), and XGBoost at the route-stop-hour level. R² values are not directly comparable across rows because each model's SST is computed over a different unit of analysis. Trip-level models face structurally higher unexplainable variance (driver, traffic, passenger noise) than aggregate models, so the lower Linear Regression R² reflects a harder prediction task, not a worse model.

*[Team: add 2–3 sentences here summarizing which model performed best for its task and the overall equity conclusion.]*

---

## Data Visualization & Results

All plots generated inline in the respective notebooks.

### XGBoost Visualizations *(Baria Mustafa)*

**1. Feature Importance (Mean Gain)**  
Horizontal bar chart of each feature's mean gain importance score, sorted ascending. Reveals where demographic features (`pct_low_income`, `pct_minority`) rank relative to operational features — the primary equity diagnostic for the XGBoost model.

<img width="884" height="584" alt="Unknown-10" src="https://github.com/user-attachments/assets/862d7fd3-1a4a-47d0-875e-b8a673c24159" />


**2. SHAP Summary Plot**  
Mean absolute SHAP values for a 5,000-row random sample of the test set. Unlike raw importances, SHAP values are additive and locally faithful — they show not just which features matter, but the direction of their effect. Comparable across models (e.g., vs. Random Forest importances).

<img width="784" height="853" alt="Unknown-11" src="https://github.com/user-attachments/assets/658b0749-3d53-4faa-8d09-9de8a5938dde" />


**3. Average Delay by Hour of Day**  
Line plot of mean and median `avg_delay_min` by hour (0–23) with ±1 std shaded band. Reveals the intra-day delay profile and whether AM/PM peaks show elevated delay. Wide std bands at peak hours indicate heterogeneous route behavior.

<img width="1084" height="484" alt="Unknown-12" src="https://github.com/user-attachments/assets/9ebc3610-b1aa-4ee4-80da-05a3967d9b18" />


**4. Adjusted R² Forward-Selection Chart**  
Two-panel: (left) CV Test R², Train R², and Adjusted R² as feature groups are added cumulatively; (right) Train-Test Gap bars (red if gap > 0.05). Directly exposes which feature groups improve generalization vs. inflate in-sample fit — the key overfitting diagnostic.

<img width="884" height="484" alt="Unknown-13" src="https://github.com/user-attachments/assets/ef146fa3-3648-48ae-b933-2c7e5403c85a" />


**5. Title VI Equity Scatter — Delay vs. Demographics**  
Side-by-side scatter plots: mean `avg_delay_min` vs. Title VI Low-Income % (left) and vs. Title VI Minority % (right), with trend lines and route ID labels. A positive slope would indicate that routes serving more low-income or minority riders experience systematically longer delays — the quantitative update to the "64 Hours" finding.

<img width="1382" height="608" alt="Unknown-14" src="https://github.com/user-attachments/assets/472d6475-cd47-430d-8949-1a09bc59d8ec" />


---
### Linear Regression Visualizations *(Primah Muwanga)*

**1. Coefficient Plot — Numeric Predictors**
Horizontal bar chart of the six numeric-predictor coefficients from the full route-FE model, color-coded by sign (red = positive, blue = negative) with values labeled in seconds of delay per unit increase. Shows `is_rush_hour` (+39.5 sec) and the joint demographic coefficients as the visually dominant effects, while operating coefficients like `hour` (+3.6) and `prev_delay` (+0.15) are smaller per unit but cumulatively meaningful across their ranges.

<img width="972" height="532" alt="Image" src="https://github.com/user-attachments/assets/6a33622e-43ae-4990-91f8-c7192997030c" />

**2. Demographics vs. Average Delay — Route-Level Scatter**
Two side-by-side scatter plots, one route per dot, with `pct_minority` and `pct_lowincome` on the x-axes and route-level mean delay on the y-axis. Red OLS fit lines with shaded 95% confidence bands. Both relationships have *negative* slopes — routes serving more minority and low-income riders have lower average delays at the route level — directly contradicting the naive "minority routes are slower" prediction. Headline equity finding for the linear regression.

<img width="1409" height="541" alt="Image" src="https://github.com/user-attachments/assets/ec80aebf-c732-437d-b81a-9b4fa66c9be5" />

**3. Residual Distribution**
Histogram of residuals (actual − predicted) with KDE overlay, x-axis clipped to ±1500 to show the bulk distribution. Residuals are roughly symmetric around zero (supporting unbiasedness of OLS estimates) but exhibit heavy tails consistent with a Laplace rather than Normal distribution. Heavy tails serve as a caveat on the parametric inference assumptions underlying OLS standard errors and p-values.

<img width="972" height="532" alt="Image" src="https://github.com/user-attachments/assets/dc03f456-08de-4b9d-8eec-2481ec9fdb1e" />

**4. Demographic Coefficients — Alone vs. Joint Fit**
Grouped bar chart comparing each demographic coefficient when fit alone (blue, with operating controls only) versus jointly with the other demographic (red). Direct evidence of multicollinearity (r = 0.815): `pct_lowincome` shifts from −504 sec/unit alone to −449 jointly, and `pct_minority` shifts from −238 alone to −37 jointly. The two variables share enough explanatory power that neither joint coefficient is interpretable in isolation — the "alone" estimates are the stable, defensible numbers and are what's reported in the Summary of Results.

<img width="972" height="477" alt="Image" src="https://github.com/user-attachments/assets/0eea894b-c680-4e40-80cb-f377a4f5f4b5" />
---

### Random Forest Visualizations *(Amira Zuniga)*

**1. Feature Importance Bar Chart**  
Horizontal bars for all 10 features sorted descending, from `route_historical_avg_delay` (0.4336) to `pct_minority` (0.0063). The central equity finding: historical delay patterns and service reliability metrics dominate prediction, while demographic features combined account for under 2% of importance — suggesting delay is structurally embedded in route operations.

<img width="1324" height="721" alt="rf_feature_importance" src="https://github.com/user-attachments/assets/fa3f81fa-8d88-4cd0-83c0-5691c7d7b05f" />

**2. Correlation Heatmap — Features and Response**  
Pearson correlation matrix across all 8 numeric features and `avg_delay`. `route_historical_avg_delay` shows the strongest correlation (r = 0.566). Demographic features show weak *negative* correlations with `avg_delay` (`pct_low_income` r = −0.105, `pct_minority` r = −0.067) — the same counter-intuitive direction seen in the linear regression, likely due to confounding with route characteristics.

<img width="1675" height="1172" alt="rf_correlation_heatmap" src="https://github.com/user-attachments/assets/dd647040-34de-436b-a4cd-f1f44065ddea" />

---

### Summary of Results

**Linear Regression** *(Primah):* The full OLS model (p=41) achieved R² = 0.0893 and RMSE = 355.42 sec, with F-statistic = 779.0 (p < 0.001), confirming at least one predictor significantly explains trip-level delay. A demographics-only model explained only 1.7% of variance; adding operating and route factors contributed an additional 7.2 percentage points. A separate partial F-test (operating + demographics vs. operating only) yielded F(2, 325917) = 1848.10, p < 0.001, with R² rising from 0.0479 to 0.0586 — rejecting the null hypothesis that demographics add no explanatory power beyond operating factors. The single-predictor refits (Visualization 5) gave the stable demographic estimates: `pct_minority` alone = −237.99 sec/unit, `pct_lowincome` alone = −504.14 sec/unit. Both negative, meaning routes serving more disadvantaged riders show *lower* average delays, a direction opposite to the naive "time tax" prediction, likely because route fixed effects absorb the structural disparity into route-level baselines.

**Random Forest** *(Amira):* Achieved the highest R² at 0.4604 (adj R² = 0.4353), RMSE = 107.37 sec, MAE = 77.47 sec on route-level data (n=226, p=10), with F-statistic = 18.34 (p < 0.001). `route_historical_avg_delay` was the dominant feature at 43.4% importance. Demographic features ranked last (<1% combined), likely reflecting absorption by route-level fixed effects rather than absence of equity disparity.

**XGBoost** *(Baria):* The XGBoost model achieved MAE = 1.796 min, RMSE = 2.750 min, and R² = 0.616 on the held-out test set (aggregated to route × stop × hour × day-of-week level). The forward adjusted-R² selection (5-fold CV, n=45,264) confirmed that route/network features and temporal features drive the bulk of generalization, and that demographic features (Title VI low-income %, minority %) meaningfully improve adjusted R² by +0.016 after controlling for all operational factors — confirming a statistically detectable equity signal. Notably, `stop_id_enc` and `hist_delay_stop` were flagged as harmful (adj R² drops, train-test gaps >0.10–0.13), indicating within-sample leakage from stop-level target encoding.

---

## Team

| Name | BU Email | Contributions |
|---|---|---|
| Baria Mustafa | bmustafa@bu.edu | XGBoost modeling pipeline, feature engineering (temporal, stop-level, demographic), adjusted R² overfitting analysis, Title VI equity visualization, SHAP interpretability |
| Primah Muwanga | pmuwanga@bu.edu | Linear regression pipeline (`merge_for_regression.py`, `lin_reg_final.py`, `regression_viz.py`); `mbta_final.csv` rebuild from arrival data + stops file; trip-level OLS with route fixed effects (n = 325,924, p = 41); naive baseline + demographics-only + full-model comparison; partial F-test for secondary-goal hypothesis; multicollinearity diagnostics with single-predictor sensitivity refits; five regression diagnostic visualizations (coefficient plot, route-level demographics scatter, predicted vs. actual, residual histogram, alone-vs-joint coefficients) | |
| Amira Zuniga | azuniga@bu.edu | *[Amira: fill in — e.g., random forest model, route-level aggregation, feature engineering, feature importance analysis, correlation heatmap]* |
