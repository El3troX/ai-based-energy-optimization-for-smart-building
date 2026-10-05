# Smart Building AI --- Project Plan

## Project

**AI-Based Energy Optimization for Smart Buildings**

## Development Strategy

Build the project incrementally. Do not attempt to generate the entire
system in one uncontrolled pass.

The project should progress through:

``` text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Occupancy ML
   ↓
Energy ML
   ↓
Optimization
   ↓
Simulation
   ↓
Dashboard
   ↓
Testing
   ↓
Documentation
```

------------------------------------------------------------------------

# Phase 0 --- Repository Setup `[COMPLETED]`

### Tasks
- [x] Create repository structure (`src/`, `notebooks/`, `data/`, `models/`, `dashboard/`, `tests/`, `docs/`).
- [x] Create Python virtual environment and `requirements.txt`.
- [x] Create `.gitignore` and initial `README.md`.
- [x] Initialize Git repository and connect to GitHub remote (`El3troX/ai-based-energy-optimization-for-smart-building`).

### Deliverable
A clean, runnable repository with reproducible environment configuration.

------------------------------------------------------------------------

# Step 0 --- Dataset Acquisition & Schema Discovery `[COMPLETED]`

### Tasks
- [x] Download full Building 59 dataset (27 raw CSVs, ~2.38 GB uncompressed) to `data/raw/`.
- [x] Inspect schemas, timestamps, missingness, and physical units across all 27 files.
- [x] Generate machine-readable inventory `docs/dataset_inventory.csv`.
- [x] Document physical relationships, sensor metadata, and CO2 non-overlap constraint in `docs/dataset.md`.

### Deliverable
`docs/dataset.md` and `docs/dataset_inventory.csv`.

------------------------------------------------------------------------

# Phase 1 --- Modeling Table Construction & Methodology Lock-In `[COMPLETED]`

### Tasks
- [x] Define spatial modeling scope: South Wing Office Zone (Floors 3 & 4).
- [x] Establish uniform 1-hour temporal resolution with physical integration ($\overline{P}_{\text{kW}} \times 1\text{ h} = \text{kWh}$).
- [x] Enforce sensor cleaning: DS18B20 error codes (`85.0°C`/`0.0°C`) and CT negative zero-drift clamping.
- [x] Implement linear interpolation for short electrical gaps $< 4$ consecutive hours; drop macro outages $\ge 4$ hours.
- [x] Implement zero-lookahead feature engineering in `src/features.py` (calendar, cyclical, thermal gradients, lags $t-k$, rolling shifted).
- [x] Enforce explicit 1-hour-ahead contiguous target shifting ($t \to t+1$): `is_occupied_next_hour`, `south_wing_total_kwh_next_hour`.
- [x] Create explicit future HVAC candidate controls: `rtu_south_fan_spd_mean_next_hour`, `rtu_south_damper_pct_mean_next_hour`.
- [x] Export validated modeling tables: `occupancy_data.parquet` (6,547 rows), `energy_data.parquet` (5,945 rows), `joint_modeling_data.parquet` (5,945 rows).
- [x] Produce comprehensive specification `docs/modeling_specification.md` and report `docs/phase1_modeling_tables.md`.

### Deliverable
Validated Parquet tables in `data/processed/` and master specifications in `docs/`.

------------------------------------------------------------------------

# Pre-ML Methodology & Leakage Audit `[PASSED]`

### Tasks
- [x] Critical Lag Audit: Validate $z_{\text{lag}\_kh} = z_{t - kh}$ on actual historical timestamps (fails if references $t, t-1, t-23$).
- [x] Rolling Window Audit: Verify rolling statistics contain ONLY historical observations $t-1 \dots t-W$.
- [x] Target Shift Audit: Verify contiguous alignment $t \to t+1$ for occupancy and energy targets.
- [x] Chained Pipeline Audit: Confirm inference requires zero ground-truth future data; maintain Oracle vs Chained evaluation modes.
- [x] Control Feature Audit: Explicitly separate observed controls at $t$ from candidate controls at $t+1$.
- [x] Documentation Audit: Synchronize all specs across `Context.md`, `Project Plan.md`, `README.md`, `docs/`.

### Deliverable
`docs/pre_ml_audit.md` and passing automated test suite `tests/test_preprocessing.py` (9/9 passed).

------------------------------------------------------------------------

# Phase 2 --- Exploratory Data Analysis `[COMPLETED]`

Created:
- `notebooks/01_eda.ipynb` (37 cells, executed in-place with all rich outputs)
- `docs/phase2_eda_report.md` (comprehensive 14-section analytical report)
- `docs/phase2_eda_validation.md` (17-item verification checklist)
- `docs/phase2_eda_metrics.json` (exact machine-readable metrics)
- `docs/figures/eda/` (16 high-resolution publication figures)

### Completed Tasks
- [x] All processed datasets loaded and quality verified (unique, monotonic, 0 nulls).
- [x] Submeter additive identity verified ($\text{Total Energy} == \text{HVAC} + \text{Lighting} + \text{Plug Loads}$ within $0.000000\text{ kWh}$).
- [x] Target distributions analyzed for all 6 targets.
- [x] Occupancy patterns characterized (diurnal profile, weekend collapse, headcount distribution).
- [x] Energy submeter profiles quantified (HVAC 78.60%, Plug Loads 14.22%, Lighting 7.18%).
- [x] Occupancy vs energy sensitivity evaluated (+657.0% lighting, +108.5% plug loads, +12.6% HVAC).
- [x] Weather and environmental thermodynamic relationships evaluated ($r = 0.691$ outdoor temp vs HVAC).
- [x] HVAC control telemetry characterized (supply fan speed, economizer damper opening).
- [x] Multi-scale temporal patterns analyzed (working-hour clustering, heatmaps).
- [x] Feature correlations and 22 collinear pairs ($|r| \ge 0.85$) identified and documented.
- [x] Chronological split timeline and seasonal distribution shift analyzed (winter temp drop).
- [x] Zero ML models trained, zero hyperparameters tuned, zero data leakage introduced.

### Deliverable
`notebooks/01_eda.ipynb`, `docs/phase2_eda_report.md`, `docs/phase2_eda_validation.md`, and 16 figures in `docs/figures/eda/`.

------------------------------------------------------------------------

# Phase 3 --- Model Training & Evaluation `[COMPLETED]`

Created:
- `src/occupancy_model.py` (data loading, candidate classifiers, headcount regression, serialization)
- `src/energy_model.py` (candidate energy regressors, chained forecasting, serialization)
- `src/evaluation.py` (classification & regression metrics, threshold tuning, publication plotting)
- `notebooks/02_occupancy_prediction.ipynb` (executed in-place with all rich outputs)
- `notebooks/03_energy_prediction.ipynb` (executed in-place with all rich outputs)
- `models/occupancy/` (Random Forest champion classifier, metadata, threshold $T^*=0.51$)
- `models/energy/` (Ridge Regression champion forecaster, scaler, metadata)
- `docs/phase3_results.json` (machine-readable experiment registry)
- `docs/phase3_model_training.md` (detailed academic training and evaluation report)
- `docs/phase3_validation.md` (20-item verification checklist)
- `tests/test_modeling.py` (8 automated tests; 17/17 project unit tests passing)
- `docs/figures/models/` (9 publication-quality diagnostic plots)

### Key Results Summary
- **Champion Occupancy Classifier**: Random Forest ($T^*=0.51$)
  - Validation F1: **0.9377** | Validation ROC-AUC: **0.9433** | Balanced Accuracy: **0.8640**
  - Test F1: **0.9170** | Test Precision: **0.9018** | Test Recall: **0.9327** | Test ROC-AUC: **0.9364**
- **Champion Energy Forecaster**: Ridge Regression ($\alpha=10$)
  - Validation RMSE (Chained): **6.2754 kWh** | Validation MAE: **5.0992 kWh** | $R^2$: **0.3843**
  - Test MAE (Oracle): **7.7061 kWh** | Test RMSE (Oracle): **9.8759 kWh**
  - Test MAE (Chained): **7.6733 kWh** | Test RMSE (Chained): **9.8357 kWh**
  - Error Propagation Penalty ($\Delta\text{MAE}$): **-0.0328 kWh** (zero degradation from Stage 1 occupancy forecast)

------------------------------------------------------------------------

# Phase 3.5 --- Forecast Robustness & Optimization Readiness Audit `[COMPLETED]`

### Tasks
- [x] Implement and benchmark naive energy baselines (Persistence $y_t$, Seasonal Naive $y_{t-24}$, lag feature).
- [x] Evaluate baselines vs Ridge champion on Validation and Test splits; export `docs/phase3_baselines.json`.
- [x] Perform forecast error decomposition by thermal regime ($<10^\circ\text{C}$, $10-16^\circ\text{C}$, $>16^\circ\text{C}$), occupancy state, working hours, and load magnitude.
- [x] Conduct HVAC component-specific error attribution (`hvac_S_kwh_next_hour`) and correlate with physical variables.
- [x] Derive control-safety occupancy threshold $T_{\text{safety}} = 0.30$ strictly from validation data (FNR $\le 1.5\%$).
- [x] Audit historical support bounds for candidate controls: fan speed $[40\%, 90\%]$, outdoor air damper $[10\%, 90\%]$.
- [x] Conduct counterfactual sensitivity sanity check: verify physical monotonicity (fan coefficient $+2.63$, damper coefficient $-0.88$).
- [x] Formulate Optimization Readiness Gate decision: **READY WITH SAFEGUARDS**.
- [x] Export master audit report `docs/phase3_robustness_audit.md` and machine-readable results `docs/phase3_robustness_results.json`.
- [x] Add automated test suite `tests/test_robustness.py` (5 tests; 22/22 total project tests passing).

### Key Results Summary
- **Baseline Comparison**:
  - Validation: Ridge beats Seasonal Naive ($\text{RMSE } 6.28\text{ vs } 7.43\text{ kWh}$, $R^2\ 0.38\text{ vs } 0.14$). Does not beat 1-step Persistence ($\text{RMSE } 4.57\text{ kWh}$, $R^2\ 0.67$), but persistence has zero control sensitivity ($\nabla_u \hat{y} = 0$).
  - Test: Ridge is competitive with Seasonal Naive ($\text{RMSE } 9.84\text{ vs } 9.88\text{ kWh}$, $R^2\ -0.06\text{ vs } -0.07$).
- **Root Cause of Test Negative $R^2$**:
  - Model trained in summer/autumn (May–Nov, mean $15.5^\circ\text{C}$) learned cooling response.
  - Winter test set (Jan–Feb, mean $9.8^\circ\text{C}$) activated space heating. In cold weather ($<10^\circ\text{C}$), model exhibits $-8.60\text{ kWh}$ bias. HVAC underprediction accounts for $76.7\%$ of total underprediction bias.
- **Safety & Control Parameters**:
  - Dual-threshold architecture: $T_{\text{class}} = 0.51$ (classification reporting), $T_{\text{safety}} = 0.30$ (control gating, $\text{FNR} = 1.50\%$, $\text{Recall} = 98.5\%$).
  - Admissible control bounds: Fan Speed $[40\%, 90\%]$, Damper Opening $[10\%, 90\%]$.
  - Physical surrogate response: Fan speed response strictly positive ($\frac{\partial E}{\partial \text{fan}} > 0$), economizer damper response strictly negative ($\frac{\partial E}{\partial \text{damper}} < 0$).
- **Gate Verdict**: **READY WITH SAFEGUARDS** (proceeding as research simulation under explicit uncertainty, bounded candidate controls, and conservative occupancy safety gating).

------------------------------------------------------------------------

# Phase 4 --- Counterfactual Optimization & Simulation `[NEXT]`

Create:
- `src/optimizer.py`
- `src/simulator.py`
- `notebooks/04_optimization_simulation.ipynb`

### Objectives
- Perform model-based counterfactual scenario search over candidate HVAC setpoints (`rtu_south_fan_spd_mean_next_hour`, `rtu_south_damper_pct_mean_next_hour`) and lighting controls within audited support bounds ($[40\%, 90\%]$ fan, $[10\%, 90\%]$ damper).
- Enforce operational and comfort constraints (ASHRAE Standard 55 thermal comfort bounds) and control-safety occupancy threshold ($T_{\text{safety}} = 0.30$).
- Simulate before/after energy and cost savings across the test partition under surrogate validity safeguards.

------------------------------------------------------------------------

# Phase 8 --- Optimization Engine

Create:

`src/optimizer.py`

## Goal

Perform **Model-Based Counterfactual Scenario Evaluation** to determine lower-energy operating conditions.
Do not claim unverified physical causality; identify valid configurations that minimize model-predicted energy under operational constraints.

### Inputs for upcoming hour $t+1$

``` text
Predicted occupancy state (is_occupied_next_hour)
Current thermal state & gradient (indoor_temp_mean, temp_gradient)
Outdoor weather conditions (outdoor_temp_c, solar_radiation)
Baseline equipment settings (lighting, RTU fan speed, damper pct)
```

### Candidate Actions & Constraints

``` text
Lighting:
    If unoccupied: Standby / OFF (drop toward standby baseline 0.29 kW)
    If occupied: ON / Normal (visual comfort constraint)

HVAC:
    If unoccupied: Setback mode (relax deadband to 65°F–78°F, reduce fan VFD speed)
    If occupied: ASHRAE Standard 55 comfort deadbands (70°F–75°F)
```

### Optimization Approach

For each valid candidate configuration:

1.  Generate feature vector for hour $t+1$.
2.  Pass candidate through trained energy model: $\hat{E}^{\text{cand}} = M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{cand}})$.
3.  Apply operational and comfort constraints (eliminate invalid states).
4.  Select valid configuration with minimum predicted energy:
    $$\hat{E}^{\text{opt}}_{t+1} = \min_{U^{\text{cand}} \in \mathcal{U}_{\text{valid}}} M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{cand}})$$
5.  Calculate estimated energy savings ($\Delta E$), cost savings ($), and CO2 reductions.

------------------------------------------------------------------------

# Phase 9 --- Building Simulation

Create:

`src/simulator.py`

The simulator should compare:

### Baseline

Current building configuration.

### Optimized

Configuration recommended by the optimizer.

Calculate:

``` text
baseline_energy
optimized_energy
energy_saved
percentage_saved
estimated_cost_saved
estimated_co2_saved
```

Formula:

``` text
energy_saved = baseline_energy - optimized_energy
```

``` text
percentage_saved =
    (energy_saved / baseline_energy) × 100
```

Cost:

``` text
cost_saved =
    energy_saved × electricity_tariff
```

CO2:

``` text
co2_saved =
    energy_saved × emission_factor
```

Clearly document the tariff and emission-factor assumptions.

------------------------------------------------------------------------

# Phase 10 --- Streamlit Dashboard

Create:

`dashboard/app.py`

## Dashboard pages/sections

### 1. Overview

Display:

-   Occupancy
-   Temperature
-   Humidity
-   Current energy
-   Predicted energy
-   Estimated savings

### 2. Energy Analytics

Charts:

-   Energy trend
-   Hourly consumption
-   Daily consumption
-   Occupancy vs energy
-   Temperature vs energy

### 3. Model Performance

Show:

-   Model comparison
-   Classification metrics
-   Regression metrics
-   Confusion matrix
-   Feature importance

### 4. Optimization Simulator

Allow the user to set:

``` text
Temperature
Humidity
Occupancy
Hour
HVAC
Lighting
```

Then run:

**OPTIMIZE BUILDING**

Show:

``` text
Current Configuration
        ↓
Current Energy

Optimized Configuration
        ↓
Optimized Energy

        ↓

Energy Saved
Cost Saved
CO2 Saved
```

### 5. Recommendations

Display human-readable recommendations such as:

``` text
Room is currently unoccupied.
Turn lighting OFF.

HVAC is operating unnecessarily.
Switch to energy-saving mode.
```

------------------------------------------------------------------------

# Phase 11 --- Testing

Create tests for:

-   Data preprocessing
-   Feature engineering
-   Model loading
-   Prediction
-   Optimization
-   Saving calculations
-   Invalid input handling

Example:

``` text
test_energy_saving_calculation()
test_optimizer_never_returns_invalid_configuration()
test_model_prediction_shape()
test_missing_input_handling()
```

------------------------------------------------------------------------

# Phase 12 --- Final Evaluation

Create a final experiment comparing:

### Occupancy Models

  Model                   Accuracy   Precision   Recall    F1   ROC-AUC
  --------------------- ---------- ----------- -------- ----- ---------
  Logistic Regression          ---         ---      ---   ---       ---
  Decision Tree                ---         ---      ---   ---       ---
  Random Forest                ---         ---      ---   ---       ---
  XGBoost                      ---         ---      ---   ---       ---

### Energy Models

  Model                 MAE   RMSE    R²
  ------------------- ----- ------ -----
  Linear Regression     ---    ---   ---
  Decision Tree         ---    ---   ---
  Random Forest         ---    ---   ---
  XGBoost               ---    ---   ---

Do not populate these tables until the models have actually been
trained.

------------------------------------------------------------------------

# Phase 13 --- Final Demonstration

The final demo should tell a simple story.

## Scenario

``` text
Time: 2:00 PM
Temperature: 29°C
Humidity: 65%
Occupancy: 0
HVAC: ON
Lights: ON
```

### Current state

``` text
Predicted Energy: 4.8 kWh
```

### Optimizer

``` text
HVAC -> Energy-saving mode
Lights -> OFF
```

### New state

``` text
Predicted Energy: 3.2 kWh
```

### Result

``` text
Energy Saved: 1.6 kWh
Percentage Saved: 33.3%
```

The actual numbers must come from the implemented model/simulation.

------------------------------------------------------------------------

# Phase 14 --- Documentation

README should contain:

1.  Project title
2.  Problem statement
3.  Motivation
4.  Objectives
5.  Architecture
6.  Dataset
7.  Feature engineering
8.  ML models
9.  Model evaluation
10. Optimization methodology
11. Dashboard
12. Results
13. Installation
14. Usage
15. Repository structure
16. Limitations
17. Future scope

------------------------------------------------------------------------

# Phase 15 --- Final Academic Material

Generate:

-   Project report
-   Architecture diagram
-   ML pipeline diagram
-   Dataset description
-   Model comparison tables
-   Results graphs
-   Dashboard screenshots
-   PPT
-   Viva questions and answers

The report should distinguish clearly between:

**Prediction**

and

**Optimization**

because they are separate components of the system.

------------------------------------------------------------------------

# Definition of Done

The project is complete when:

-   [ ] Dataset is documented.
-   [ ] EDA is complete.
-   [ ] Preprocessing is reproducible.
-   [ ] Feature engineering is implemented.
-   [ ] Occupancy model is trained and evaluated.
-   [ ] Energy model is trained and evaluated.
-   [ ] Multiple models are compared.
-   [ ] Best models are selected based on actual results.
-   [ ] Models are saved and reloadable.
-   [ ] Optimization engine is implemented.
-   [ ] Baseline vs optimized simulation works.
-   [ ] Energy/cost/CO2 savings are calculated.
-   [ ] Streamlit dashboard works.
-   [ ] Tests exist for critical components.
-   [ ] README is complete.
-   [ ] No fabricated results exist.
-   [ ] Project can be explained clearly in a viva.

------------------------------------------------------------------------

# Suggested Future Scope

Possible extensions:

-   Real IoT sensor integration.
-   Live MQTT data.
-   Reinforcement learning for HVAC control.
-   Multi-room optimization.
-   Dynamic electricity pricing.
-   Weather API integration.
-   Carbon-aware optimization.
-   Cloud deployment.
-   Edge inference.
-   Digital twin simulation.
-   Reinforcement-learning-based building control.
