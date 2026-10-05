# Phase 1 Validation Report: Modeling Table Construction

**Document Version:** 1.0  
**Phase:** Phase 1 — Modeling Table Construction  
**Author:** Lead ML Engineer & Software Architect  
**Project:** AI-Based Energy Optimization for Smart Buildings  
**Target File Location:** `docs/phase1_modeling_tables.md`  

---

## 1. Executive Summary

Phase 1 establishes the reproducible data processing pipeline that transforms raw heterogeneous telemetry from Building 59 into standardized, leakage-free modeling tables at **hourly (`1h`) resolution**.

The pipeline is implemented in modular production scripts:
- [src/preprocessing.py](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/src/preprocessing.py): Loads raw files, applies sensor-level cleaning, performs temporal resampling, and exports validated Parquet/CSV tables.
- [src/features.py](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/src/features.py): Generates calendar indicators, cyclical trigonometric coordinates, thermodynamic gradients, strictly causal lag features ($\text{lag} \ge 1$), and strictly causal rolling window statistics.
- [notebooks/01_modeling_table_construction.ipynb](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/notebooks/01_modeling_table_construction.ipynb): Interactive pipeline verification and visualization notebook.

---

## 2. Verified Spatial & Temporal Scope

- **Spatial Boundary:** **South Wing Office Zone** (Flors 3 & 4), directly aligned with the TRAF-SYS overhead stereo-vision camera counters (`occ_third_south`, `occ_fourth_south`) and the submetered electrical channels (`lig_S`, `mels_S`, `hvac_S`).
- **Temporal Benchmark Horizon:**
  $$\mathbf{2018\text{-}05\text{-}22\text{ 07:00:00}\quad\text{to}\quad 2019\text{-}02\text{-}21\text{ 10:00:00}}$$
  (275 consecutive days / 6,604 initial potential hourly intervals).

---

## 3. Sensor Cleaning Rules Applied

1. **DS18B20 Temperature Logger Fault Cleansing:**  
   In `zone_temp_interior.csv`, hardware power-on reset codes (`85.0°C`), ground disconnects (`0.0°C`), and out-of-range physical readings ($T < 12^\circ\text{C}$ or $T > 38^\circ\text{C}$) are replaced with `np.nan` and linearly interpolated over short transient intervals before spatial averaging.
2. **Current Transducer Zero-Drift Clamping:**  
   In `ele.csv`, slight negative power readings (-0.01 to -0.44 kW) caused by sensor zero-offset drift during off-hours are clamped to `0.0`.
3. **Duplicate Timestamp Resolution:**  
   Raw multi-rate files containing timestamp duplicates (e.g. event triggers in RTU logs) are grouped by timestamp and averaged prior to hourly resampling.
4. **Physical Energy Conversion:**  
   In an hourly framework ($\Delta t = 1\text{ h}$), average active power in kilowatts ($\overline{P}_{\text{kW}}$) directly equals active energy consumption in **kilowatt-hours (kWh)**:
   $$\text{Energy (kWh)} = \overline{P}_{\text{kW}} \times 1\text{ h}$$

---

## 4. Modeling Table Specifications

### 4.1 Occupancy Modeling Table (`occupancy_data.parquet`)

- **File Path:** [data/processed/occupancy_data.parquet](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/data/processed/occupancy_data.parquet) (also available as `.csv`)
- **Row Count:** **6,544 rows**
- **Column Count:** **37 columns** (1 timestamp + 2 targets + 34 features)
- **Time Range:** `2018-05-23 07:00:00` to `2019-02-21 10:00:00`
- **Missingness After Preprocessing:** **0 missing values (0.0%)**
- **Rows Removed:** 60 rows (0.91% of window, due to initial 24-hour lag requirement)
- **Chronological Monotonicity:** Verified `True`
- **Timestamp Uniqueness:** Verified `True`

#### Schema Breakdown:
- **Index/Key:** `timestamp`
- **Targets:**
  1. `is_occupied` (int, {0, 1}): Binary presence ($1$ if peak hourly headcount $> 0$, else $0$). Class balance: 56.1% occupied, 43.9% unoccupied.
  2. `occ_total_mean` (float, $\ge 0$): Hourly mean occupant headcount.
- **Features (34):**
  - *Indoor Environment:* `indoor_temp_mean`, `indoor_temp_min`, `indoor_temp_max`, `indoor_temp_diff_1h`, `temp_gradient_in_out`
  - *Outdoor Meteorology:* `outdoor_temp_c`, `relative_humidity`, `dew_point_temp_c`, `solar_radiation`
  - *HVAC Operational Status:* `rtu_south_fan_spd_mean`, `rtu_south_damper_pct_mean`
  - *Calendar & Schedule:* `hour`, `day_of_week`, `is_weekend`, `is_business_hour`, `month`
  - *Cyclical Trigonometric Coordinates:* `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
  - *Strictly Causal Past Lags:* `occ_total_mean_lag_1h`, `occ_total_mean_lag_2h`, `occ_total_mean_lag_24h`, `is_occupied_lag_1h`, `is_occupied_lag_2h`, `is_occupied_lag_24h`
  - *Strictly Causal Rolling Statistics:* `occ_total_mean_rolling_mean_3h`, `occ_total_mean_rolling_std_3h`, `occ_total_mean_rolling_mean_6h`, `occ_total_mean_rolling_std_6h`, `occ_total_mean_rolling_mean_24h`, `occ_total_mean_rolling_std_24h`

---

### 4.2 Energy Modeling Table (`energy_data.parquet`)

- **File Path:** [data/processed/energy_data.parquet](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/data/processed/energy_data.parquet) (also available as `.csv`)
- **Row Count:** **5,951 rows**
- **Column Count:** **45 columns** (1 timestamp + 4 targets + 40 features)
- **Time Range:** `2018-05-23 07:00:00` to `2019-02-21 10:00:00`
- **Missingness After Preprocessing:** **0 missing values (0.0%)**
- **Rows Removed:** 653 rows (9.89% of window, due to initial 24h lag and unrecorded submeter communications)
- **Retention Rate:** **90.11%**
- **Chronological Monotonicity:** Verified `True`
- **Timestamp Uniqueness:** Verified `True`

#### Schema Breakdown:
- **Index/Key:** `timestamp`
- **Targets:**
  1. `south_wing_total_kwh` (float): Total South Wing electricity consumption (kWh).
  2. `lig_S_kwh` (float): Overhead lighting electricity consumption (kWh).
  3. `mels_S_kwh` (float): Plug-load electricity consumption (kWh).
  4. `hvac_S_kwh` (float): HVAC equipment electricity consumption (kWh).
- **Features (40):**
  - *Occupancy State (Predictors):* `is_occupied`, `occ_total_mean`
  - *Indoor Environment:* `indoor_temp_mean`, `indoor_temp_diff_1h`, `temp_gradient_in_out`
  - *Outdoor Meteorology:* `outdoor_temp_c`, `relative_humidity`, `dew_point_temp_c`, `solar_radiation`
  - *HVAC Controllable Operations:* `rtu_south_fan_spd_mean`, `rtu_south_damper_pct_mean`
  - *Calendar & Schedule:* `hour`, `day_of_week`, `is_weekend`, `is_business_hour`, `month`
  - *Cyclical Coordinates:* `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
  - *Strictly Causal Energy Lags:* `south_wing_total_kwh_lag_1h`, `south_wing_total_kwh_lag_2h`, `south_wing_total_kwh_lag_24h`, `lig_S_kwh_lag_1h`, `lig_S_kwh_lag_2h`, `lig_S_kwh_lag_24h`, `mels_S_kwh_lag_1h`, `mels_S_kwh_lag_2h`, `mels_S_kwh_lag_24h`, `hvac_S_kwh_lag_1h`, `hvac_S_kwh_lag_2h`, `hvac_S_kwh_lag_24h`
  - *Strictly Causal Rolling Statistics:* `south_wing_total_kwh_rolling_mean_3h`, `south_wing_total_kwh_rolling_std_3h`, `south_wing_total_kwh_rolling_mean_6h`, `south_wing_total_kwh_rolling_std_6h`, `south_wing_total_kwh_rolling_mean_24h`, `south_wing_total_kwh_rolling_std_24h`

---

## 5. Strict Data Leakage Audit

| Leakage Risk Category | Implementation Defense | Status |
|---|---|---|
| **Future Observation Leakage** | All lag features enforce $\text{lag} \ge 1$. Rolling features shift the time series by $1$ step (`shift(1)`) before computing rolling statistics. | **VERIFIED (Zero Lookahead)** |
| **Concurrent Target Leakage in Occupancy** | The occupancy table strictly excludes all concurrent energy submeter measurements. | **VERIFIED (No Energy Features in Occupancy Table)** |
| **Component-Sum Additive Leakage** | Concurrent submeter readings (`lig_S_kwh`, `mels_S_kwh`, `hvac_S_kwh`) are strictly banned from predicting `south_wing_total_kwh`. Only historical lags are used. | **VERIFIED (No Additive Leakage)** |
| **Lookahead Transformation Bias** | All feature scalers/encoders will be fitted strictly on the training partition ($t \le \text{2018-11-30}$). | **READY FOR PHASE 3/4** |

---

## 6. Assumptions & Limitations

1. **Spatial Aggregation:** South Wing office space covers Floors 3 & 4. Occupancy headcounts are combined to match the spatial circuit boundary of the South Wing electrical submeters.
2. **Short Telemetry Gaps:** Small electrical telemetry gaps ($< 4$ consecutive hours) were linearly interpolated. Large continuous blocks were dropped.
3. **CO2 Exclusion:** As proven in Step 0, `zone_co2.csv` telemetry begins in August 2019, 6 months after camera occupancy concluded. CO2 cannot be used in camera occupancy modeling.
4. **Airflow Sensor Dropouts:** Supply airflow sensors (`rtu_sa_fr.csv`) suffered substantial physical hardware outages in 2018. The robust, 100% complete actuator signals (`rtu_south_fan_spd_mean` and `rtu_south_damper_pct_mean`) are used as the primary HVAC operational controls, preserving over 90% of observations.
