# Machine Learning & Optimization Modeling Specification

**Document Version:** 1.0 (Methodology Lock-In)  
**Status:** Frozen & Approved for Implementation  
**Project:** AI-Based Energy Optimization for Smart Buildings  
**Author:** Lead ML Engineer & Software Architect  
**Location:** `docs/modeling_specification.md`  

---

## 1. Spatial Scope & System Boundaries

- **Physical Facility:** Building 59 (Wang Hall), Lawrence Berkeley National Laboratory (LBNL), Berkeley, California (ASHRAE Climate Zone 3C).
- **Primary Modeling Boundary:** **South Wing Office Zone** across the 3rd and 4th floors.
- **Physical Rationale:**
  - Overhead TRAF-SYS stereo-vision thermal/optical counting cameras (`occ.csv`) were installed exclusively at the entrance portals to the South Wing office zones of Floors 3 & 4.
  - Electrical submetering (`ele.csv`) cleanly partitions electrical distribution panels into South Wing circuits:
    - `lig_S`: Overhead lighting circuits for the South Wing.
    - `mels_S`: Miscellaneous electrical loads (plug-loads / desktop equipment) for the South Wing.
    - `hvac_S`: Dedicated terminal unit fans and mechanical conditioning for the South Wing.
  - Rooftop Units **RTU 003** and **RTU 004** directly serve the South Wing thermal zones.
  - Confining the modeling boundary to the South Wing ensures strict physical alignment between occupant sensors, submeters, and air distribution equipment.

---

## 2. Temporal Resolution & Physical Units

- **Modeling Time Step:** **1 Hour ($\Delta t = 1\text{ h}$)**.
- **Resampling Frequency:** `freq='1h'` (hourly intervals aligned on the hour: `00:00`, `01:00`, ..., `23:00`).
- **Physical Unit Equivalence:**
  In an hourly discretization framework, the mathematical integration of average active electrical power $\overline{P}_{\text{kW}}$ over 1 hour directly yields electrical energy in kilowatt-hours (kWh):
  $$E_{\text{hourly}} = \int_{t}^{t+1} P(\tau)\, d\tau \approx \overline{P}_{\text{kW}} \times 1\text{ h} = \text{kWh}$$
- **Data Types:**
  - Energy & Submeters: Active energy consumption in **kWh**.
  - Temperatures: Degrees Celsius (**°C**).
  - Solar Irradiance: Global Horizontal Irradiance (**W/m²**).
  - Humidity: Relative humidity percentage (**%**).
  - Occupancy Headcount: Mean continuous occupant count over the hour.
  - Occupancy State: Binary discrete state $\{0, 1\}$.

---

## 3. Explicit Prediction Horizon

The system strictly solves a **1-Hour-Ahead Predictive Forecasting Task** ($H = +1\text{ hour}$), not a concurrent regression task.

### Mathematical Formulation:
At timestamp $t$ (end of current hour), the system observes all sensor inputs up to and including $t$:
$$X_t = \{x_t, x_{t-1}, x_{t-2}, \dots\}$$

The pipeline forecasts future states for the upcoming hour $t+1$:
1. **Occupancy Model ($M_{\text{occ}}$):**
   $$\hat{y}^{\text{occ}}_{t+1} = M_{\text{occ}}(X_t)$$
2. **Energy Model ($M_{\text{energy}}$):**
   $$\hat{y}^{\text{energy}}_{t+1} = M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U_{t+1})$$
   where $U_{t+1}$ represents planned or candidate HVAC/lighting operational control settings for hour $t+1$.

### Formal Target Variable Names:
- `is_occupied_next_hour`: Binary indicator of occupancy presence during hour $t+1$.
- `occ_total_mean_next_hour`: Continuous mean occupant headcount during hour $t+1$.
- `south_wing_total_kwh_next_hour`: Total South Wing electrical energy (kWh) consumed during hour $t+1$.
- `lig_S_kwh_next_hour`: Lighting electrical energy (kWh) consumed during hour $t+1$.
- `mels_S_kwh_next_hour`: Plug-load electrical energy (kWh) consumed during hour $t+1$.
- `hvac_S_kwh_next_hour`: HVAC electrical energy (kWh) consumed during hour $t+1$.

---

## 4. Target Definitions

### 4.1 Occupancy Prediction
1. **Binary Classification (Primary):**
   $$\text{is\_occupied\_next\_hour} = \begin{cases} 1 & \text{if } \max(\text{occ\_total}_{\tau}) > 0 \text{ for } \tau \in (t, t+1] \\ 0 & \text{otherwise} \end{cases}$$
   - **Class Balance:** 56.1% Occupied, 43.9% Unoccupied (balanced, avoiding severe class imbalance issues).
   - **Evaluation Metrics:** F1-score, Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix.
2. **Headcount Regression (Secondary):**
   $$\text{occ\_total\_mean\_next\_hour} = \frac{1}{60} \sum_{m=1}^{60} (\text{occ\_third\_south}_m + \text{occ\_fourth\_south}_m)$$
   - **Evaluation Metrics:** MAE, RMSE, $R^2$.

### 4.2 Energy Prediction
1. **Whole-Zone Energy Regression (Primary):**
   $$\text{south\_wing\_total\_kwh\_next\_hour} = \text{lig\_S\_kwh\_next\_hour} + \text{mels\_S\_kwh\_next\_hour} + \text{hvac\_S\_kwh\_next\_hour}$$
   - **Evaluation Metrics:** MAE, RMSE, $R^2$, MAPE (where safe).
2. **Multi-End-Use Submeter Regression (Diagnostic / Optimization):**
   - Separate models or multi-output regression for `lig_S_kwh_next_hour`, `mels_S_kwh_next_hour`, and `hvac_S_kwh_next_hour`.

---

## 5. Feature Availability, Lags & Rolling Window Rules

### 5.1 Temporal Feature Availability
- Calendar and diurnal features for hour $t$ and hour $t+1$ are deterministically known in advance (calendar math):
  - `hour`, `day_of_week`, `is_weekend`, `is_business_hour`, `month`
  - Cyclical coordinates: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, etc.

### 5.2 Strictly Causal Lag Definitions
For any time series $z_t$, a lag feature of order $k$ is defined strictly as:
$$z_{\text{lag}\_kh} = z_{t - kh} \quad \text{relative to } t \quad (k \ge 1)$$
- For $k=1$: $z_{\text{lag}\_1h} = z_{t - 1h}$ (the observation at the previous hour).
- For $k=2$: $z_{\text{lag}\_2h} = z_{t - 2h}$ (the observation two hours prior).
- For $k=24$: $z_{\text{lag}\_24h} = z_{t - 24h}$ (the observation at the exact same hour on the previous day).
- Minimum lag step is $1$ hour. Zero or negative lags ($k \le 0$) are strictly prohibited in predictor matrices.

### 5.3 Strictly Causal Rolling Window Definitions
Rolling statistics (mean, standard deviation) over window $W \in \{3, 6, 24\}$ hours are computed strictly on shifted past observations:
$$\text{rolling\_mean}_W(z)_t = \frac{1}{W} \sum_{i=1}^{W} z_{t - i}$$
- For $W=3$: $\text{rolling\_mean}_{3h}(z)_t = \frac{z_{t-1} + z_{t-2} + z_{t-3}}{3}$
- For $W=6$: $\text{rolling\_mean}_{6h}(z)_t = \frac{z_{t-1} + z_{t-2} + \dots + z_{t-6}}{6}$
- For $W=24$: $\text{rolling\_mean}_{24h}(z)_t = \frac{z_{t-1} + z_{t-2} + \dots + z_{t-24}}{24}$
By applying `shift(1)` prior to `.rolling(W)`, concurrent observations at $t$ are excluded from the rolling window, eliminating lookahead leakage.

### 5.4 Feature Availability Matrix

| Feature Name | Category | Available at $t$ | Used in Occupancy Model | Used in Energy Model |
|---|---|---|---|---|
| `indoor_temp_mean` | Indoor Thermal | Yes ($t$) | Yes | Yes |
| `indoor_temp_diff_1h` | Thermal Rate of Change | Yes ($t$) | Yes | Yes |
| `temp_gradient_in_out` | Thermodynamic Gradient | Yes ($t$) | Yes | Yes |
| `outdoor_temp_c` | Weather | Yes ($t$) | Yes | Yes |
| `relative_humidity` | Weather | Yes ($t$) | Yes | Yes |
| `solar_radiation` | Weather | Yes ($t$) | Yes | Yes |
| `rtu_south_fan_spd_mean` | Observed HVAC Controls | Yes ($t$) | Yes | Yes |
| `rtu_south_damper_pct_mean`| Observed HVAC Controls | Yes ($t$) | Yes | Yes |
| `calendar / cyclical` | Temporal | Yes (deterministic) | Yes | Yes |
| `occ_total_mean_lag_1/2/24h`| Causal Lag | Yes ($t-1, t-2, t-24$) | Yes | No (captured via energy lags) |
| `is_occupied_lag_1/2/24h` | Causal Lag | Yes ($t-1, t-2, t-24$) | Yes | No |
| `is_occupied_next_hour` | Occupancy at $t+1$ | **Predicted** by $M_{\text{occ}}$ | **TARGET** | **PREDICTOR** |
| `occ_total_mean_next_hour` | Headcount at $t+1$ | **Predicted** by $M_{\text{occ}}$ | **TARGET** | **PREDICTOR** |
| `rtu_south_fan_spd_mean_next_hour` | Planned/Candidate Control at $t+1$ | **Candidate Action** from Optimizer | No | **PREDICTOR / CONTROL** |
| `rtu_south_damper_pct_mean_next_hour` | Planned/Candidate Control at $t+1$ | **Candidate Action** from Optimizer | No | **PREDICTOR / CONTROL** |
| `south_wing_total_kwh_lag_1/2/24h`| Causal Lag | Yes ($t-1, t-2, t-24$) | **No (Leakage)** | Yes |
| `lig_S_kwh_lag_1/2/24h` | Causal Lag | Yes ($t-1, t-2, t-24$) | **No (Leakage)** | Yes |
| `hvac_S_kwh_lag_1/2/24h`| Causal Lag | Yes ($t-1, t-2, t-24$) | **No (Leakage)** | Yes |
| `mels_S_kwh_lag_1/2/24h`| Causal Lag | Yes ($t-1, t-2, t-24$) | **No (Leakage)** | Yes |

---

## 6. Occupancy $\rightarrow$ Energy Chained Inference Pipeline

In real-world deployment, ground-truth occupancy at $t+1$ cannot be observed before hour $t+1$ occurs. Therefore, the pipeline operates in two stages:

```text
               Sensed Data at time t
         (Indoor Climate, Weather, HVAC, Lags)
                         │
                         ▼
             Occupancy Forecast Model
                    (M_occ)
                         │
                         ▼
        Predicted Occupancy State at t+1
    (is_occupied_next_hour / occ_total_next_hour)
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
Energy Forecasting Model          Optimization Engine
      (M_energy)                  (Counterfactual Search)
         │                               │
         ▼                               ▼
Predicted Baseline Energy        Optimal Setpoint Recommendations
       at t+1                       for Lighting & HVAC at t+1
```

### Academic Training & Evaluation Strategy:
To maintain full empirical rigor, we evaluate both:
1. **Oracle Energy Model (Upper Bound):**
   The Energy Model is evaluated using historical ground-truth occupancy at $t+1$. This isolates the pure regression capability of the energy architecture assuming perfect occupancy knowledge.
2. **Chained / Stacked Pipeline (Real-World Deployment):**
   The Energy Model is evaluated on the holdout test set using the *out-of-sample predicted occupancy* $\hat{y}^{\text{occ}}_{t+1}$ generated by $M_{\text{occ}}$.
   - This quantifies error propagation from occupancy misclassifications to energy prediction errors.
   - It guarantees that published test performance reflects actual operational conditions.

---

## 7. Preprocessing & Gap Interpolation Rules

To ensure both data completeness and physical fidelity without introducing synthetic distortion, the pipeline enforces strict, verified preprocessing rules:

### 7.1 Electrical Submeter Gap Interpolation Rule
- **The Rule:**
  - **Short electrical gaps $< 4$ consecutive hours** ($< 16$ consecutive missing 15-minute readings, or up to 3 consecutive missing hours in resampled space) **may be linearly interpolated**.
  - **Long electrical gaps $\ge 4$ consecutive hours** must **NEVER be interpolated**. They are preserved as missing blocks (`NaN`) and strictly dropped from training and evaluation tables.
- **Physical & Operational Rationale:**
  1. *Physical Autocorrelation & Baseload Continuity:* Commercial office buildings exhibit strong electrical and thermal inertia over short sub-4-hour horizons. Base electrical loads (baseload IT servers, desktop plug-loads, standby emergency lighting, steady HVAC minimum ventilation) vary smoothly within 1- to 3-hour windows. Linear interpolation across isolated 1- to 3-hour telemetry dropouts introduces negligible error while preserving contiguous time-series continuity essential for causal lag computation ($t-1, t-2, t-24$).
  2. *Sensor Packet Drops vs. Physical Outages:* Short sub-4-hour telemetry dropouts in Building 59 represent transient gateway buffer overruns, Modbus communication timeouts, or momentary logger reboots rather than genuine physical equipment shutdowns.
  3. *Distortion Prevention on Macro Outages:* Outages exceeding 4 hours (e.g. multi-day network disconnects such as November 16–28, 2018 or August 20–23, 2018) span complete diurnal cycles, weekday-to-weekend transitions, or operational mode shifts. Linear interpolation across 4+ hours would synthesize fictitious flat or linear power profiles, corrupting diurnal dynamics and introducing severe data distortion. Dropping these intervals guarantees that the ML models train only on physically authentic building behavior.

### 7.2 Indoor Temperature Sensor Cleaning
- In `zone_temp_interior.csv`, readings of `85.0°C` (the well-documented DS18B20 digital temperature sensor power-on reset state) and `0.0°C` (ground disconnect) are replaced with `np.nan`.
- Physical bounds: Any interior reading outside $[12.0^\circ\text{C}, 38.0^\circ\text{C}]$ is treated as a sensor fault and filtered before spatial averaging across the South Wing loggers.

### 7.3 Electrical Current Transducer (CT) Zero-Drift Clamping
- Slight negative readings in `mels_S` and `lig_S` (ranging from $-0.01$ to $-0.44\text{ kW}$) during night hours are physical zero-calibration offsets under zero current. They are clamped to `0.0 kW` (`np.clip(val, a_min=0, a_max=None)`).

### 7.4 Multi-Rate Resampling to Hourly Resolution
- All disparate sensor sampling intervals (1-minute camera counts, 1-minute RTU controls, 10-minute Raspberry Pi interior temperatures, and 15-minute submeters and weather data) are resampled to uniform 1-hour intervals (`freq='1h'`) using physically grounded aggregation operators:
  - Energy/Power: Hourly mean active power ($\overline{P}_{\text{kW}} \times 1\text{ h} = \text{kWh}$).
  - Temperature/Weather/HVAC: Hourly arithmetic mean.
  - Occupancy: Hourly mean headcount (`occ_total_mean`) and hourly maximum presence indicator (`is_occupied`).

---

## 8. Joint Modeling Dataset Construction

To support unified multi-stage training, evaluation, and before/after simulation, an explicit **Joint Modeling Table** is constructed:
- **Path:** `data/processed/joint_modeling_data.parquet` (and `.csv`)
- **Row Count:** **5,945 rows**
- **Column Count:** **61 columns** (All features and targets from both domains synchronized on identical timestamps)
- **Time Range:** `2018-05-23 07:00:00` to `2019-02-21 09:00:00`

### Dropped Rows Audit:

| Table | Rows | Dropped from Initial Overlap (6,604) | Reason for Drop |
|---|---|---|---|
| `occupancy_data.parquet` | 6,547 | 57 rows (0.86%) | 24-hour lag initialization window + final forecasting boundary shift ($t \to t+1$). |
| `energy_data.parquet` | 5,945 | 659 rows (9.98%) | Initial 24h lag requirement + 479 hours of raw submeter power communication outages $\ge 4\text{ h}$ (e.g. Nov 16–28, Aug 20–23) + lag re-initialization rows. |
| `joint_modeling_data.parquet` | 5,945 | 602 rows relative to occupancy | Exact intersection: dropped rows correspond strictly to the electrical meter communication outages $\ge 4\text{ h}$. |

---

## 9. Chronological Train / Validation / Test Strategy

Because building operations exhibit strong seasonality and autocorrelation, random shuffling (K-Fold cross-validation) causes catastrophic temporal data leakage and is strictly prohibited.

The benchmark dataset is partitioned strictly chronologically:

```text
|====================== TRAIN ======================|===== VAL =====|===== TEST =====|
2018-05-23                                       2018-11-30      2019-01-10       2019-02-21
                  (~70% / 4,160 rows)              (~15% / 890 rows)  (~15% / 895 rows)
```

1. **Training Partition (70%):** `2018-05-23 07:00:00` to `2018-11-30 23:00:00` (Late Spring, Summer, Autumn).
2. **Validation Partition (15%):** `2018-12-01 00:00:00` to `2019-01-10 23:00:00` (Early Winter; used for hyperparameter tuning and model selection).
3. **Test Partition (15%):** `2019-01-11 00:00:00` to `2019-02-21 09:00:00` (Mid Winter; strictly held out for final evaluation).

### Preprocessing Leakage Rule:
- All scalers (e.g. `StandardScaler`, `MinMaxScaler`) and any data transformations must be **fitted strictly on the Training partition**.
- Preprocessing objects must be saved (`joblib`) and applied to Validation and Test partitions using `.transform()` only.

---

## 10. Optimization Engine Semantics

The Optimization Engine operates as a **Model-Based Counterfactual Scenario Evaluator**.

### Academic & Physical Disclaimer:
- The optimizer evaluates counterfactual scenarios using the surrogate ML model.
- It does **not** claim causal or physical certainty beyond the predictive domain of the trained model.
- Estimated energy, cost, and CO2 reductions represent *model-predicted potential savings* under specified supervisory control adjustments.

### Counterfactual Optimization Formulation:
For each upcoming hour $t+1$:
1. **Inputs:**
   - Predicted occupancy state $\hat{y}^{\text{occ}}_{t+1}$.
   - Exogenous weather conditions ($T^{\text{oa}}_t$, Solar irradiance).
   - Current thermal state ($T^{\text{in}}_t$).
   - Current baseline equipment settings ($U^{\text{base}}_{t+1}$: Lighting state, HVAC fan speed, temperature setpoints).
2. **Candidate Actions ($U^{\text{cand}}$):**
   - **Lighting Control:**
     - If $\hat{y}^{\text{occ}}_{t+1} = 0$ (unoccupied): Recommend `Lights = OFF / Standby` (drops lighting power to standby baseline of 0.29 kW, eliminating non-essential illumination).
     - If $\hat{y}^{\text{occ}}_{t+1} = 1$ (occupied): Enforce visual comfort constraint (`Lights = ON / Normal`).
   - **HVAC Setpoint & Fan Control:**
     - If unoccupied: Relax cooling/heating temperature deadbands (setback mode: widen setpoint band from $70^\circ\text{F}\text{--}74^\circ\text{F}$ to $65^\circ\text{F}\text{--}78^\circ\text{F}$), reducing supply fan VFD speed (`rtu_south_fan_spd`).
     - If occupied: Maintain ASHRAE Standard 55 thermal comfort deadbands ($21^\circ\text{C}\text{--}24^\circ\text{C}$ / $70^\circ\text{F}\text{--}75^\circ\text{F}$).
3. **Evaluation & Selection:**
   $$\hat{E}^{\text{base}}_{t+1} = M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{base}}_{t+1})$$
   $$\hat{E}^{\text{opt}}_{t+1} = \min_{U^{\text{cand}} \in \mathcal{U}_{\text{valid}}} M_{\text{energy}}(X_t, \hat{y}^{\text{occ}}_{t+1}, U^{\text{cand}})$$
   $$\Delta E_{t+1} = \max\left(0, \hat{E}^{\text{base}}_{t+1} - \hat{E}^{\text{opt}}_{t+1}\right)$$
4. **Impact Metrics:**
   - **Cost Savings ($):** $\Delta E \times \text{Tariff}$ (e.g. PG&E Commercial Time-Of-Use rate, peak vs. off-peak $/kWh).
   - **CO2 Reductions ($\text{kg CO}_2$):** $\Delta E \times \text{Emission Factor}$ (California eGRID regional marginal emission factor: $0.22\text{ kg CO}_2/\text{kWh}$).
