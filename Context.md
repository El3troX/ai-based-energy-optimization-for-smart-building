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
Occupancy + Environment + HVAC + Historical Energy + Weather
                         |
                         v
                  Machine Learning
                         |
              +----------+----------+
              |                     |
              v                     v
      Occupancy Prediction   Energy Prediction
              |                     |
              +----------+----------+
                         |
                         v
                 Optimization
                         |
                         v
              Building Energy Savings
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

------------------------------------------------------------------------

## 3. Intended System Architecture

``` text
Smart Building Dataset / Sensor Data
                |
                v
       Data Preprocessing
                |
                v
       Feature Engineering
                |
       +--------+---------+
       |                  |
       v                  v
Occupancy Model     Energy Model
Classification       Regression
       |                  |
       +--------+---------+
                |
                v
        Optimization Engine
                |
       +--------+---------+
       |        |         |
       v        v         v
      HVAC   Lighting   Devices
                |
                v
       Before/After Simulation
                |
                v
       Streamlit Dashboard
```

------------------------------------------------------------------------

## 4. Machine Learning Components

### 4.1 Occupancy Prediction

Treat occupancy prediction as a classification problem.

Potential input features:

-   Temperature
-   Humidity
-   CO2
-   Light intensity
-   Hour
-   Day of week
-   Weekend indicator
-   Previous occupancy
-   Previous environmental readings
-   HVAC state
-   Lighting state

Target:

``` text
occupied = 0 or 1
```

Candidate algorithms:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   XGBoost, if appropriate

Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion Matrix

Do not rely on accuracy alone if the dataset is imbalanced.

------------------------------------------------------------------------

### 4.2 Energy Consumption Prediction

Treat energy consumption prediction as a regression problem.

Potential input features:

-   Occupancy
-   Temperature
-   Humidity
-   CO2
-   HVAC usage
-   Lighting usage
-   Hour
-   Day of week
-   Weekend/weekday
-   Previous energy consumption
-   Rolling energy statistics
-   Outdoor/environmental conditions when available

Target:

``` text
energy_consumption
```

Candidate algorithms:

-   Linear Regression
-   Ridge Regression
-   Decision Tree Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor
-   XGBoost Regressor, if appropriate

Metrics:

-   MAE
-   MSE
-   RMSE
-   R²
-   MAPE when appropriate and numerically safe

------------------------------------------------------------------------

## 5. Optimization Layer

The optimization layer should be clearly separated from the ML models.

ML predicts what is likely to happen.

The optimization layer decides what should be changed.

Example:

``` text
Current state:
Occupancy = 0
Temperature = 29°C
HVAC = ON
Lights = ON

Optimization:
HVAC -> Energy-saving mode
Lights -> OFF

Predicted current consumption = 4.8 kWh
Predicted optimized consumption = 3.2 kWh

Estimated saving = 1.6 kWh
```

The optimizer can initially be rule-based or simulation-based. A more
advanced implementation can use constrained optimization.

Possible optimization objectives:

``` text
Minimize:
    Energy Consumption
```

subject to constraints such as:

``` text
Occupancy comfort
Temperature bounds
Maximum HVAC changes
Required lighting level
Device availability
```

Do not claim that the optimization algorithm itself is machine learning
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
