# Smart Building AI --- Project Context

## Project Title

**AI-Based Energy Optimization for Smart Buildings**

### Core Idea

Build an end-to-end machine learning system that predicts building
occupancy and energy consumption, then uses those predictions to
recommend energy-efficient operating decisions for HVAC, lighting, and
other energy-consuming devices.

The project should behave like a **smart-building energy management
system**, not merely an energy-consumption prediction notebook.

------------------------------------------------------------------------

## 0. Selected Dataset

### Primary Dataset

**Three-Year Building Operational Performance Dataset — Building 59**

Source: Dryad / Scientific Data dataset associated with the Building 59 operational performance study.

The project will use Building 59 as the primary dataset rather than a generic energy-consumption dataset.

### Why This Dataset Was Selected

The dataset is well aligned with the project because it contains a combination of:

- Whole-building energy measurements
- End-use energy measurements
- HVAC operational data
- Indoor environmental measurements
- Outdoor/environmental measurements
- Occupancy information
- Long-term chronological observations
- A large collection of sensors and meters

The dataset provides substantially more depth than a simple energy-forecasting dataset and supports the project's intended pipeline:

```text
Current Observations at time t (Environment, Weather, HVAC, Lags)
                         │
                         ▼
             Occupancy Forecast Model (M_occ)
                         │
                         ▼
        Predicted Occupancy at t+1 (is_occupied_next_hour)
                         │
                         ▼
              Energy Forecast Model (M_energy)
                         │
                         ▼
      Predicted Energy at t+1 (south_wing_total_kwh_next_hour)
                         │
                         ▼
       Optimization Engine (Counterfactual Scenario Search)
                         │
                         ▼
         Energy & Cost-Optimized Setpoint Recommendations
```

### Dataset Handling Rule

Do not assume column names, units, sampling frequencies, or relationships between files.

**Step 0 must be dataset inspection.**

Before implementing the ML pipeline:

1. Download the dataset.
2. Inspect all available files.
3. Build a data dictionary.
4. Identify timestamps and sampling frequencies.
5. Identify occupancy variables.
6. Identify energy variables and end uses.
7. Identify HVAC variables.
8. Identify indoor environmental variables.
9. Identify outdoor/weather variables.
10. Determine which files can be joined safely.
11. Check missingness and data quality.
12. Define the final modeling tables.
13. Only then freeze the feature/target design.

Never fabricate a feature that does not exist in the source data. If a required variable is unavailable, document the limitation or redesign that component.

---

## 1. Problem Statement

Buildings consume significant amounts of electricity through HVAC
systems, lighting, and other electrical equipment. A large portion of
this consumption can be reduced when equipment operation is aligned with
occupancy and environmental conditions.

The goal of this project is to develop an intelligent system that can:

1.  Understand occupancy patterns.
2.  Predict energy consumption.
3.  Identify the environmental and operational factors driving
    consumption.
4.  Recommend energy-efficient device settings.
5.  Estimate potential energy, cost, and CO2 savings.
6.  Present the results through an interactive dashboard.

The system should use historical/simulated smart-building data and
machine learning models to make data-driven decisions.

------------------------------------------------------------------------

## 2. Project Objectives

### Primary Objectives

-   Perform exploratory data analysis on smart-building data.
-   Preprocess sensor and energy-consumption data.
-   Engineer temporal, environmental, occupancy, and lag-based features.
-   Build an occupancy classification model.
-   Build an energy-consumption regression model.
-   Compare multiple ML algorithms.
-   Select the best-performing models using appropriate evaluation
    metrics.
-   Build an optimization layer on top of the ML predictions.
-   Simulate building operation before and after optimization.
-   Quantify energy and cost savings.
-   Develop an interactive Streamlit dashboard.

### Secondary Objectives

-   Provide model explainability.
-   Visualize energy-consumption trends.
-   Identify important factors affecting energy consumption.
-   Allow users to modify building conditions interactively.
-   Provide actionable optimization recommendations.

-------------------------------------------------## 3. Intended System Architecture

The system implements a sequential, chained predictive architecture that mirrors real-time BEMS supervisory control:

``` text
Smart Building Sensed Data at time t (Indoor Temp, Weather, HVAC Controls, Lags)
                                  │
                                  ▼
                         Data Preprocessing
                                  │
                                  ▼
                        Feature Engineering
                                  │
                                  ▼
                       Occupancy Model (M_occ)
                     Classification / Regression
                                  │
                                  ▼
                Predicted Future Occupancy at t+1
          (is_occupied_next_hour, occ_total_mean_next_hour)
                                  │
                                  ▼
                         Energy Model (M_energy)
                        Multi-End-Use Regression
                                  │
                                  ▼
                  Predicted Future Energy at t+1
                 (south_wing_total_kwh_next_hour)
                                  │
                                  ▼
             Optimization Engine (Counterfactual Search)
       (Evaluates candidate HVAC/lighting settings via M_energy)
                                  │
                                  ▼
                       Before/After Simulation
                                  │
                                  ▼
                         Streamlit Dashboard
```

------------------------------------------------------------------------

## 4. Machine Learning Components

### 4.1 Occupancy Prediction (Forecasting Horizon: $t \to t+1$)

Treat occupancy prediction as a 1-hour-ahead forecasting problem ($H = +1\text{ hour}$): using observations available at timestamp $t$, forecast occupancy during the upcoming hour $t+1$.

Input features available at time $t$:
- Indoor temperatures & rate of change (`indoor_temp_mean`, `indoor_temp_diff_1h`, `temp_gradient_in_out`)
- Outdoor meteorology (`outdoor_temp_c`, `relative_humidity`, `dew_point_temp_c`, `solar_radiation`)
- HVAC operational status (`rtu_south_fan_spd_mean`, `rtu_south_damper_pct_mean`)
- Calendar & schedule (`hour`, `day_of_week`, `is_weekend`, `is_business_hour`, `month`)
- Cyclical trigonometric encodings (`hour_sin`, `hour_cos`, `day_of_week_sin`, etc.)
- Causal historical occupancy lags (`is_occupied_lag_1h/2h/24h`, `occ_total_mean_lag_1h/2h/24h`)
- Causal rolling window statistics (`occ_total_mean_rolling_mean_3h/6h/24h`, rolling std)

*Note on CO2:* In Building 59, `zone_co2.csv` telemetry begins in August 2019, 6 months after camera occupancy tracking concluded (February 2019). Therefore, CO2 is physically non-concurrent with camera occupancy and is excluded from occupancy models to prevent an empty dataset.

Explicit Targets:
1. **Primary Binary Classification:**
   ``` text
   is_occupied_next_hour = 0 or 1
   ```
   (Class balance: 56.1% occupied, 43.9% unoccupied).
2. **Secondary Headcount Regression:**
   ``` text
   occ_total_mean_next_hour >= 0.0
   ```

Candidate algorithms:
- Logistic Regression (interpretable baseline)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Evaluation Metrics:
- Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix, Accuracy.

------------------------------------------------------------------------

### 4.2 Energy Consumption Prediction (Forecasting Horizon: $t \to t+1$)

Treat energy consumption prediction as a 1-hour-ahead regression problem: using observations up to hour $t$, forecast South Wing electricity consumption during the upcoming hour $t+1$.

Input features:
- **Future Occupancy:** Occupancy state for hour $t+1$ (see evaluation strategy below).
- Indoor environmental state at $t$ (`indoor_temp_mean`, `indoor_temp_diff_1h`, `temp_gradient_in_out`)
- Outdoor meteorology at $t$ (`outdoor_temp_c`, `relative_humidity`, `dew_point_temp_c`, `solar_radiation`)
- Controllable HVAC operational settings (`rtu_south_fan_spd_mean`, `rtu_south_damper_pct_mean`)
- Calendar & cyclical indicators (`hour`, `day_of_week`, `is_weekend`, cyclical coordinates)
- Causal historical energy lags (`south_wing_total_kwh_lag_1h/2h/24h`, submeter lags)
- Causal rolling energy statistics (`south_wing_total_kwh_rolling_mean_3h/6h/24h`, rolling std)

Explicit Targets:
``` text
south_wing_total_kwh_next_hour   (Primary Whole-Zone Energy in kWh)
lig_S_kwh_next_hour              (Submeter: Lighting Energy in kWh)
mels_S_kwh_next_hour             (Submeter: Plug-Loads in kWh)
hvac_S_kwh_next_hour             (Submeter: HVAC Equipment in kWh)
```

Candidate algorithms:
- Linear / Ridge Regression (interpretable baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluation Metrics:
- MAE, RMSE, R², MAPE (where numerically safe).

### Academic Evaluation Strategy (Chained Pipeline):
To ensure scientific rigor and reflect real deployment conditions:
1. **Oracle Energy Model (Upper Bound):**
   Trained and evaluated with historical ground-truth occupancy at $t+1$. This establishes the maximum regression performance assuming perfect occupancy foresight.
2. **Chained Inference Pipeline (Real Deployment):**
   Evaluated on the holdout test set using out-of-sample predicted occupancy $\hat{y}^{\text{occ}}_{t+1}$ generated by $M_{\text{occ}}$. This benchmarks error propagation from occupancy predictions into energy forecasts.

------------------------------------------------------------------------

## 5. Optimization Layer

The optimization layer performs **Model-Based Counterfactual Scenario Evaluation**.

### Academic & Physical Disclaimer:
- The optimizer evaluates counterfactual operating scenarios using the trained surrogate energy model $M_{\text{energy}}$.
- It does **not** assert unverified physical causality; rather, it identifies control configurations that minimize *model-predicted energy* while strictly honoring operational and comfort constraints.

### Optimization Workflow:
For each upcoming hour $t+1$:
1. **Receive Context:** Current thermal/weather state at $t$, predicted occupancy $\hat{y}^{\text{occ}}_{t+1}$, and baseline control settings $U^{\text{base}}_{t+1}$.
2. **Generate Candidate Configurations ($U^{\text{cand}}$):**
   - **Lighting:**
     - Unoccupied ($\hat{y}^{\text{occ}}_{t+1} = 0$): Recommend `Lights = OFF / Standby` (reducing power toward standby baseline of 0.29 kW).
     - Occupied ($\hat{y}^{\text{occ}}_{t+1} = 1$): Enforce visual comfort constraint (`Lights = ON / Normal`).
   - **HVAC:**
     - Unoccupied: Apply temperature setback (relax deadband from $70^\circ\text{F}\text{--}74^\circ\text{F}$ to $65^\circ\text{F}\text{--}78^\circ\text{F}$), reducing RTU fan VFD speed.
     - Occupied: Maintain ASHRAE Standard 55 thermal comfort deadbands ($21^\circ\text{C}\text{--}24^\circ\text{C}$ / $70^\circ\text{F}\text{--}75^\circ\text{F}$).
3. **Pass Candidates Through Trained Energy Model:**
   Evaluate predicted energy $\hat{E}^{\text{cand}}_{t+1} = M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{cand}})$.
4. **Apply Operational & Comfort Constraints:**
   Eliminate candidates that violate temperature, ventilation, or equipment safety bounds.
5. **Select Optimal Configuration:**
   Select the valid configuration with the lowest predicted energy:
   $$\hat{E}^{\text{opt}}_{t+1} = \min_{U^{\text{cand}} \in \mathcal{U}_{\text{valid}}} M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{cand}})$$
   $$\text{Estimated Savings} = \max\left(0, \hat{E}^{\text{base}}_{t+1} - \hat{E}^{\text{opt}}_{t+1}\right)$$
6. **Quantify Financial & Environmental Impact:**
   - Cost Savings ($) = $\text{Savings (kWh)} \times \text{TOU Tariff (\$/kWh)}$
   - CO2 Reductions ($\text{kg CO}_2$) = $\text{Savings (kWh)} \times 0.22\text{ kg CO}_2/\text{kWh}$ (California eGRID regional factor)rning
unless it actually is.

------------------------------------------------------------------------

## 6. Dashboard Requirements

The dashboard should contain:

### Overview

-   Current occupancy
-   Current temperature
-   Current humidity
-   Current energy consumption
-   Predicted energy consumption
-   Estimated savings

### Energy Analytics

-   Energy consumption over time
-   Hourly consumption
-   Daily consumption
-   Consumption by device/system
-   Occupied vs unoccupied consumption

### ML Performance

-   Model comparison
-   Regression metrics
-   Classification metrics
-   Confusion matrix
-   Feature importance
-   SHAP plots if implemented

### Optimization

Interactive inputs such as:

-   Temperature
-   Humidity
-   Occupancy
-   Hour
-   HVAC state
-   Lighting state
-   Device states

Button:

**Optimize Building**

Output:

-   Current energy estimate
-   Optimized energy estimate
-   Energy saved
-   Percentage saved
-   Estimated cost saved
-   Estimated CO2 reduction
-   Recommended device actions

------------------------------------------------------------------------

## 7. Technology Direction

Preferred stack:

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost where useful
-   Matplotlib
-   Seaborn
-   Streamlit
-   Joblib
-   SHAP, optionally

The project should remain understandable and reproducible.

Avoid unnecessary technologies unless they provide a clear project
benefit.

------------------------------------------------------------------------

## 8. Engineering Principles

-   Keep data processing separate from model training.
-   Keep model training separate from inference.
-   Keep optimization separate from prediction.
-   Avoid a single giant Python file.
-   Use reusable functions.
-   Store trained models separately.
-   Use configuration files where appropriate.
-   Make random seeds reproducible.
-   Prevent data leakage.
-   Use time-aware splitting for time-series prediction where
    appropriate.
-   Save preprocessing artifacts alongside models when required.
-   Validate inputs before inference.
-   Provide useful error messages.
-   Keep notebooks for exploration and `src/` for reusable production
    code.

------------------------------------------------------------------------

## 9. Dataset-Specific Modeling Direction

The selected Building 59 dataset should support two principal supervised-learning tasks.

### Occupancy Prediction

Predict occupancy using available environmental, temporal, historical, and operational signals.

The exact target must be determined after dataset inspection.

Possible target formulations include:

- Occupied vs unoccupied classification
- Occupancy count regression
- Occupancy category classification

Do not choose the target until the actual occupancy variables have been inspected.

### Energy Consumption Prediction

Predict future or current energy consumption using available:

- Occupancy
- Environmental conditions
- HVAC operation
- End-use measurements
- Temporal features
- Weather variables
- Lagged energy observations

The exact prediction horizon and target meter must be determined from the dataset structure.

### Multi-End-Use Analysis

Where data quality permits, separately analyze:

- Whole-building energy
- HVAC energy
- Lighting energy
- Plug-load energy
- Other available end uses

This will make the optimization layer more meaningful because recommendations can be tied to specific energy-consuming systems.

---

## 10. Dataset Integrity and Leakage Prevention

Because Building 59 contains chronological building data, time-series leakage is a major concern.

Follow these rules:

- Preserve chronological order.
- Do not randomly mix future observations into training data.
- Create train/validation/test periods chronologically.
- Ensure lag and rolling features only use information available before the prediction timestamp.
- Do not use future occupancy, future energy, or future sensor values as predictors.
- Fit scalers/encoders only on training data.
- Document every target and feature timestamp relationship.

---

## 11. Expected Repository Structure

``` text
smart-building-ai/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_occupancy_prediction.ipynb
│   ├── 03_energy_prediction.ipynb
│   └── 04_optimization.ipynb
|
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── occupancy_model.py
│   ├── energy_model.py
│   ├── optimizer.py
│   ├── simulator.py
│   └── evaluation.py
|
├── models/
|
├── dashboard/
│   └── app.py
|
├── tests/
|
├── docs/
|
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

## 12. Important Academic Constraints

This is an ML academic project.

The final implementation must make it possible to explain:

-   Why the dataset was selected.
-   What each feature represents.
-   Why each model was selected.
-   How preprocessing was performed.
-   How train/test splitting was performed.
-   How hyperparameters were selected.
-   Why one model performed better than another.
-   Which metrics were used and why.
-   How the optimization layer works.
-   How energy savings are calculated.
-   What assumptions were made.
-   What limitations exist.

Never fabricate experimental results.

Any performance number in the README, report, or presentation must come
from an actual experiment.

------------------------------------------------------------------------

## 13. Desired End Result

The finished project should feel like a small **AI-powered Building
Energy Management System (BEMS)**.

The key narrative is:

``` text
Sense
  ->
Understand
  ->
Predict
  ->
Optimize
  ->
Measure Savings
```

The strongest demonstration is a before/after simulation showing how the
recommended control strategy reduces energy consumption while respecting
occupancy and comfort constraints.

------------------------------------------------------------------------

## 14. Phase Completion & Current Roadmap Status

- **Phase 0 (Repository Architecture):** COMPLETED
- **Step 0 (Dataset Discovery):** COMPLETED
- **Phase 1 (Modeling Tables $t \to t+1$):** COMPLETED
- **Pre-ML Leakage Audit:** PASSED (18/18 checks)
- **Phase 2 (Exploratory Data Analysis):** COMPLETED
- **Phase 3 (Model Training & Evaluation):** COMPLETED
- **Phase 3.5 (Robustness & Optimization Readiness):** COMPLETED (Gate: Ready with Safeguards)
- **Phase 4 (Counterfactual Optimization & Simulation):** COMPLETED (Gate: Passed with Safeguards)
- **Phase 5 (Interactive Streamlit BEMS Dashboard):** COMPLETED (8 Pages, 41/41 Tests Passing)

**FULL PROJECT IMPLEMENTATION ROADMAP IS 100% COMPLETE.**
