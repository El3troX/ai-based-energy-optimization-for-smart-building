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

# Phase 0 --- Repository Setup

### Tasks

-   Create repository structure.
-   Create Python virtual environment.
-   Create `requirements.txt`.
-   Create `.gitignore`.
-   Create initial README.
-   Create `src/`, `notebooks/`, `data/`, `models/`, `dashboard/`, and
    `tests/`.

### Deliverable

A clean, runnable repository with no ML implementation yet.

------------------------------------------------------------------------

# Phase 1 --- Dataset Selection

## Goal

Find a public smart-building dataset suitable for both occupancy
prediction and energy prediction.

### Dataset requirements

Prefer a dataset containing as many of these as possible:

-   Timestamp
-   Energy consumption
-   Occupancy
-   Temperature
-   Humidity
-   CO2
-   Lighting
-   HVAC/device information

If a single dataset cannot support every component, document the
limitation instead of fabricating variables.

### Tasks

1.  Evaluate candidate datasets.
2.  Record source and license.
3.  Download the selected dataset.
4.  Place the original file in `data/raw/`.
5.  Document columns and units.
6.  Determine prediction targets.

### Deliverable

`docs/dataset.md`

------------------------------------------------------------------------

# Phase 2 --- Exploratory Data Analysis

Create:

`notebooks/01_eda.ipynb`

### Analyze

-   Dataset dimensions
-   Data types
-   Missing values
-   Duplicate records
-   Outliers
-   Target distributions
-   Time trends
-   Occupancy distribution
-   Energy distribution
-   Correlation matrix
-   Temperature vs energy
-   Occupancy vs energy
-   Hour vs energy
-   Weekday vs weekend consumption

### Visualizations

At minimum:

-   Energy over time
-   Occupancy over time
-   Hourly energy profile
-   Occupied vs unoccupied energy
-   Temperature vs energy
-   Correlation heatmap
-   Energy distribution

### Deliverable

A documented EDA notebook with conclusions.

------------------------------------------------------------------------

# Phase 3 --- Data Preprocessing

Create reusable functions in:

`src/preprocessing.py`

### Tasks

-   Handle missing values.
-   Remove or justify duplicates.
-   Handle invalid readings.
-   Detect extreme outliers.
-   Parse timestamps.
-   Sort chronologically.
-   Encode categorical variables.
-   Scale features where required.
-   Create reproducible preprocessing pipelines.

### Important

Do not fit transformations on the entire dataset before splitting.

Preprocessing must avoid data leakage.

### Deliverable

Clean preprocessing pipeline.

------------------------------------------------------------------------

# Phase 4 --- Feature Engineering

Create:

`src/features.py`

### Temporal features

-   Hour
-   Minute where useful
-   Day of week
-   Month
-   Weekend
-   Working hours

### Lag features

Where supported:

-   Previous energy consumption
-   Previous occupancy
-   Previous temperature

### Rolling features

Where appropriate:

-   Rolling mean energy
-   Rolling mean temperature
-   Rolling occupancy

### Interaction features

Potential examples:

``` text
occupancy × temperature
occupancy × HVAC
hour × occupancy
```

Only keep features that are justified and do not introduce leakage.

### Deliverable

Reusable feature-engineering pipeline.

------------------------------------------------------------------------

# Phase 5 --- Occupancy Prediction

Create:

`notebooks/02_occupancy_prediction.ipynb`

and reusable code in:

`src/occupancy_model.py`

### Models

Start with:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest

Optionally:

4.  Gradient Boosting
5.  XGBoost

### Evaluation

Calculate:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion matrix

### Model selection

Select the final model based on the actual validation results.

Do not automatically select Random Forest or XGBoost just because they
are more advanced.

### Deliverables

-   Trained model
-   Evaluation results
-   Comparison table
-   Confusion matrix
-   Feature importance where applicable

------------------------------------------------------------------------

# Phase 6 --- Energy Consumption Prediction

Create:

`notebooks/03_energy_prediction.ipynb`

and:

`src/energy_model.py`

### Models

Start with:

1.  Linear Regression
2.  Decision Tree Regressor
3.  Random Forest Regressor

Optionally:

4.  Gradient Boosting
5.  XGBoost

### Evaluation

Calculate:

-   MAE
-   RMSE
-   R²
-   MAPE when appropriate

### Important

If the data is chronological, use a time-aware train/validation/test
split rather than randomly shuffling future observations into the
training set.

### Deliverables

-   Model comparison table
-   Actual vs predicted plot
-   Residual analysis
-   Feature importance
-   Saved final model

------------------------------------------------------------------------

# Phase 7 --- Explainability

Optional but recommended.

Use:

-   Feature importance
-   Permutation importance
-   SHAP

Answer:

> What factors cause the model to predict higher energy consumption?

Potential findings might involve:

-   Occupancy
-   HVAC
-   Temperature
-   Hour
-   Lighting

Only report findings supported by the trained model and data.

------------------------------------------------------------------------

# Phase 8 --- Optimization Engine

Create:

`src/optimizer.py`

## Goal

Use predictions to determine lower-energy operating conditions.

### Input

``` text
occupancy
temperature
humidity
time
HVAC state
lighting state
device states
```

### Candidate actions

``` text
HVAC:
    ON
    OFF
    ENERGY_SAVING

Lighting:
    ON
    OFF
    DIMMED
```

### Basic optimization approach

For each valid candidate configuration:

1.  Generate the feature vector.
2.  Predict energy consumption.
3.  Apply comfort/operational constraints.
4.  Calculate estimated cost.
5.  Select the lowest-energy valid configuration.

Conceptually:

``` text
For each possible configuration:
    predicted_energy = energy_model(configuration)

    if configuration satisfies constraints:
        keep configuration

Select configuration with minimum predicted_energy
```

This creates an interpretable optimization layer around the ML model.

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
