# Dataset Investigation Report: Building 59 Operational Performance Dataset

**Document Version:** 1.0  
**Phase:** Step 0 — Dataset Acquisition, Inspection & Schema Discovery  
**Author:** Lead ML Engineer & Software Architect  
**Project:** AI-Based Energy Optimization for Smart Buildings  
**Target File Location:** `docs/dataset.md`  

---

## 1. Dataset Overview

The primary dataset for this project is the **Three-Year Building Operational Performance Dataset — Building 59** (Wang Hall), constructed in 2015 at the Lawrence Berkeley National Laboratory (LBNL) in Berkeley, California. 

Building 59 is a four-story, research and office building with a total and conditioned floor area of approximately **111,904 ft² (10,396 m²)** located in ASHRAE Climate Zone 3C (warm-summer Mediterranean climate). The building is served by:
- A variable air volume (VAV) rooftop air handling system (4 Rooftop Units: RTU 001–004) with air-source heat pumps (ASHP) for space heating/cooling.
- Under-floor air distribution (UFAD) terminal units with variable fan speeds and hydronic reheat coils.
- High-efficiency LED lighting and Energy Star-compliant plug loads.
- Submetered electrical circuits separating HVAC, lighting, and miscellaneous plug loads for the North and South wings.
- A research-grade sensor network deployed across two primary office floors (3rd and 4th floors), including overhead stereo-vision camera occupancy counters, environmental loggers (Raspberry Pi Zero W + DS18B20 temperature sensors), and outdoor on-site weather stations.

The data was collected over a three-year span (**January 1, 2018 to December 31, 2020**) and curated by LBNL researchers using a multi-algorithm data-cleaning pipeline (linear interpolation, K-nearest neighbors, matrix factorization) to address raw telemetry dropouts.

---

## 2. Source & Official Citation

- **Repository:** Dryad Digital Repository
- **Dataset DOI:** [https://doi.org/10.7941/D1N33Q](https://doi.org/10.7941/D1N33Q)
- **Open Access License:** Creative Commons Zero v1.0 (CC0 1.0) Universal / Public Domain
- **Official Journal Reference:**
  > Luo, N., Wang, Z., Blum, D., Bourassa, N., Broughton, J., Weyandt, C., Piette, M. A., & Hong, T. (2022). *A three-year dataset supporting research on building energy management and occupancy analytics*. **Scientific Data**, 9(1), 157. DOI: `10.1038/s41597-022-01257-x`.
- **Supporting Technical Repository:** [https://github.com/LBNL-ETA/Data-Cleaning](https://github.com/LBNL-ETA/Data-Cleaning)
- **Semantic Metadata:** Brick Schema Model (`Bldg59_w_occ Brick model.ttl`) & Semantic Metadata JSON (`Bldg59_w_occ metadata of dataset.json`).

---

## 3. File Inventory

The raw dataset located in `data/raw/` consists of:
1. `Bldg59_clean data/`: 27 time-series CSV files (~2.38 GB uncompressed)
2. `Bldg59_w_occ Brick model.ttl`: Brick RDF ontology defining building equipment hierarchy and sensor topology.
3. `Bldg59_w_occ metadata of dataset.json`: Machine-readable metadata schema.
4. `data_description_table_3year_clean_data.xlsx`: Data dictionary detailing sensor codes, sampling rates, and historical raw missingness.
5. `README_Dryad_Bldg59.txt`: Experimental setup and data curation overview.
6. `metadata_Dryad_Bldg59.docx`: Dataset narrative documentation.

### Complete Inventory of the 27 Time-Series CSV Files

| File Name | Domain Category | Size (MB) | Row Count | Col Count | Nominal Step | Start Timestamp | End Timestamp | Dup Timestamps | Missing Values |
|---|---|---|---|---|---|---|---|---|---|
| `ele.csv` | Energy Submeters | 5.88 | 103,048 | 7 | 15 min | 2018-01-01 01:00 | 2021-01-01 00:00 | 0 | 71,092* |
| `occ.csv` | Occupancy Ground Truth | 11.12 | 396,193 | 3 | 1 min | 2018-05-22 07:00 | 2019-02-21 10:12 | 0 | 0 |
| `wifi.csv` | Occupancy Proxy | 2.61 | 101,047 | 5 | 5 min | 2018-05-22 00:00 | 2020-12-31 23:55 | 12 | 14,688 |
| `site_weather.csv` | Outdoor Meteorology | 5.07 | 105,217 | 6 | 15 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 0 | 0 |
| `zone_temp_interior.csv` | Indoor Environment | 31.13 | 149,696 | 17 | 10 min | 2018-02-22 00:30 | 2021-01-01 00:00 | 0 | 63,167 |
| `zone_temp_exterior.csv` | Indoor Environment | 401.40 | 1,511,904 | 52 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 34,456 | 5,213,819 |
| `zone_co2.csv` | Indoor Air Quality | 37.22 | 565,481 | 12 | 1 min | 2019-08-19 01:33 | 2021-01-01 00:00 | 11,763 | 0 |
| `zone_temp_sp_c.csv` | HVAC Setpoints | 38.85 | 237,133 | 52 | 5 min | 2018-09-15 10:00 | 2021-01-01 00:00 | 0 | 1,401,623 |
| `zone_temp_sp_h.csv` | HVAC Setpoints | 37.77 | 237,126 | 52 | 5 min | 2018-09-15 10:00 | 2021-01-01 00:00 | 0 | 1,400,521 |
| `rtu_econ_sp.csv` | HVAC Setpoints | 39.28 | 1,384,223 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_sa_p_sp.csv` | HVAC Setpoints | 54.75 | 1,407,300 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 34 | 41,875 |
| `rtu_sa_t_sp.csv` | HVAC Setpoints | 74.40 | 1,578,292 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_fan_spd.csv` | HVAC Operation | 82.56 | 1,382,051 | 9 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 0 | 0 |
| `rtu_oa_damper.csv` | HVAC Operation | 45.18 | 1,384,223 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_oa_fr.csv` | HVAC Operation | 78.48 | 1,384,223 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_sa_fr.csv` | HVAC Operation | 92.08 | 1,407,353 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 34 | 42,596 |
| `rtu_ma_t.csv` | HVAC Operation | 47.71 | 1,382,778 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_oa_t.csv` | HVAC Operation | 47.72 | 1,382,778 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_ra_t.csv` | HVAC Operation | 47.68 | 1,382,778 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 51 | 0 |
| `rtu_sa_t.csv` | HVAC Operation | 56.45 | 1,407,323 | 5 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 34 | 42,041 |
| `rtu_plenum_p.csv` | HVAC Operation | 131.90 | 1,381,902 | 9 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 0 | 0 |
| `uft_fan_spd.csv` | HVAC Operation | 381.03 | 1,511,261 | 52 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 33,484 | 5,196,780 |
| `uft_hw_valve.csv` | HVAC Operation | 440.77 | 2,072,154 | 45 | 1 min | 2018-01-01 00:00 | 2021-01-01 00:00 | 594,708 | 4,601,623 |
| `hp_hws_temp.csv` | HVAC Operation | 22.15 | 1,048,575 | 2 | 1 min | 2018-01-01 00:00 | 2020-05-02 11:57 | 4 | 0 |
| `ashp_meter.csv` | Plant Equipment | 1.43 | 43,978 | 2 | 5 min | 2020-08-01 07:05 | 2020-12-31 23:50 | 0 | 0 |
| `ashp_hw.csv` | Plant Equipment | 5.29 | 124,487 | 4 | 5 min | 2019-10-26 18:05 | 2020-12-31 23:50 | 1 | 0 |
| `ashp_cw.csv` | Plant Equipment | 1.65 | 43,979 | 4 | 5 min | 2020-08-01 07:00 | 2020-12-31 23:50 | 0 | 0 |

*\*Note on `ele.csv` missingness:* 67,912 of the missing entries belong to `Unnamed: 6`, which represents an additional submeter column activated only in 2020. The primary end-use channels have <1.5% missingness.

---

## 4. Variable Semantics & Data Dictionary

### 4.1 Electrical Energy Submeters (`ele.csv`)
Sampling interval: 15 minutes. Unit: **kW** (average active electrical demand over 15 min).

| Column | Physical Meaning | Wing / Zone | Mean (kW) | Min (kW) | Max (kW) | Missing (%) |
|---|---|---|---|---|---|---|
| `mels_S` | Miscellaneous Electrical Loads (Plug loads) | South Wing | 2.57 | -0.34* | 14.58 | 0.04% |
| `lig_S` | Overhead Lighting System Power | South Wing | 1.46 | -0.05* | 10.71 | 0.03% |
| `hvac_S` | HVAC Equipment Power | South Wing | 19.72 | 0.00 | 79.80 | 1.50% |
| `mels_N` | Miscellaneous Electrical Loads (Plug loads) | North Wing | 8.75 | -0.44* | 35.98 | 0.02% |
| `hvac_N` | HVAC Equipment Power | North Wing | 21.23 | 0.00 | 84.50 | 1.50% |
| `Unnamed: 6` | Supplemental 2020 Metered Load | Whole/Other | 18.96 | 0.00 | 64.81 | 65.9% (2018-19 absent) |

*\*Slight negative values (-0.05 to -0.44 kW) are telemetry baseline drift at near-zero loads; must be clamped to 0 during preprocessing.*

### 4.2 Occupancy Ground Truth (`occ.csv`)
Sampling interval: 1 minute. Sourced from overhead TRAF-SYS stereo-vision thermal/optical counting cameras positioned at portals to the south open-plan office zones.

| Column | Physical Meaning | Wing / Zone | Mean (count) | Median | Max (count) | % Zero (Unoccupied) | % Positive (Occupied) |
|---|---|---|---|---|---|---|---|
| `occ_third_south` | Instantaneous occupant count | 3rd Floor South | 6.59 | 1.0 | 74.0 | 44.0% | 56.0% |
| `occ_fourth_south` | Instantaneous occupant count | 4th Floor South | 4.85 | 1.0 | 60.0 | 47.5% | 52.5% |

### 4.3 Outdoor Meteorological Conditions (`site_weather.csv`)
Sampling interval: 15 minutes. Sourced from Synopticlabs weather station on LBNL campus. Zero missing values across the 3 years.

| Column | Physical Meaning | Unit | Mean | Min | Max | Standard Dev |
|---|---|---|---|---|---|---|
| `air_temp_set_1` | Outdoor dry-bulb air temperature (Sensor 1) | °C | 14.24 | -0.33 | 42.33 | 4.88 |
| `air_temp_set_2` | Outdoor dry-bulb air temperature (Sensor 2) | °C | 13.81 | -0.56 | 46.11 | 4.97 |
| `dew_point_temperature_set_1d` | Outdoor dew point temperature | °C | 6.86 | -14.61 | 18.92 | 4.32 |
| `relative_humidity_set_1` | Outdoor relative humidity | % | 66.10 | 6.40 | 95.50 | 18.73 |
| `solar_radiation_set_1` | Global horizontal solar irradiance | W/m² | 189.61 | 0.00 | 1101.00 | 273.45 |

### 4.4 Indoor Climate (`zone_temp_interior.csv`)
Sampling interval: 10 minutes. Sourced from 16 calibrated DS18B20 digital temperature sensors logged via Raspberry Pi Zero W units across open office and private office zones.

| Variable Group | Meaning | Unit | Typical Range | Known Sensor Artifact |
|---|---|---|---|---|
| `cerc_templogger_1` to `16` | Indoor air dry-bulb temperature | °C | 18.0°C – 26.5°C | Reading of `85.0°C` is the DS18B20 power-on reset state; `0.0°C` indicates disconnection. Must be cleaned as sensor faults. |

### 4.5 Rooftop Air Handling Units (`rtu_*.csv`)
Sampling interval: 1 minute.
- `rtu_fan_spd.csv`: Supply fan (`sf_vfd_spd_fbk_tn`) and return fan (`rf_vfd_spd_fbk_tn`) speeds (0–100%).
- `rtu_oa_damper.csv`: Outdoor air economizer damper position `oadmpr_pct` (0–100%).
- `rtu_sa_fr.csv` / `rtu_oa_fr.csv`: Supply and outdoor air volumetric flow rates in CFM (0 to ~12,000 CFM).
- `rtu_sa_t.csv` / `rtu_oa_t.csv` / `rtu_ma_t.csv` / `rtu_ra_t.csv`: Supply, outdoor, mixed, and return air temperatures (°F).
- `rtu_sa_t_sp.csv`: Supply air temperature setpoint (°F, controlled around 55°F–65°F).

---

## 5. Timestamp & Sampling Frequency Analysis

The dataset features heterogeneous sampling frequencies and differing temporal horizons across sensors:

```text
Time Horizon & Overlap:
--------------------------------------------------------------------------------------
2018-01-01                2018-05-22          2019-02-21                    2021-01-01
    |                          |                   |                             |
    |==========================|===================|=============================|  ele.csv (15m)
    |==========================|===================|=============================|  site_weather.csv (15m)
    |==========================|===================|=============================|  rtu_*.csv (1m)
    |              |===========|===================|=============================|  zone_temp_interior (10m)
    |                          |===================|                             |  occ.csv (1m) [9 MONTHS]
    |                          |                   |       |=====================|  zone_co2.csv (1m)
--------------------------------------------------------------------------------------
```

### Critical Sampling Discoveries:
1. **The Core ML Alignment Window:**  
   Ground-truth camera occupancy (`occ.csv`) is recorded continuously from **May 22, 2018 07:00:00 to February 21, 2019 10:12:00** (275 consecutive days / ~9 months).
2. **Synchronous Sensor Availability:**  
   During this 275-day period, `ele.csv` (15m), `site_weather.csv` (15m), `zone_temp_interior.csv` (10m), and `rtu_*.csv` (1m) are **all active and concurrent**.
3. **The CO2 Non-Overlap Discrepancy:**  
   `zone_co2.csv` begins on **August 19, 2019**, which is 6 months *after* camera occupancy tracking concluded. Therefore, **CO2 cannot be used as an input feature to predict ground-truth camera occupancy**. Any pipeline requiring CO2 for occupancy training on this dataset would result in an empty set.

---

## 6. Empirical Correlation & Physical Relationships

Analyzing the resampled joint data during the 275-day benchmark window demonstrates striking physical relationships:

### Correlation Matrix (Hourly Resolution)

| | Occupied (`is_occ`) | Occ Count (`occ_mean`) | Lighting (`lig_S`) | Plug Load (`mels_S`) | HVAC (`hvac_S`) | Total South Power | Outdoor Temp (`T_oa`) | Indoor Temp (`T_in`) |
|---|---|---|---|---|---|---|---|---|
| **Occupied (`is_occ`)** | 1.000 | 0.380 | 0.475 | 0.382 | 0.092 | 0.228 | 0.096 | 0.240 |
| **Occ Count (`occ_mean`)** | 0.380 | 1.000 | **0.682** | **0.822** | 0.200 | **0.442** | 0.196 | 0.235 |
| **Lighting (`lig_S`)** | 0.475 | **0.682** | 1.000 | **0.820** | 0.126 | **0.426** | 0.124 | 0.299 |
| **Plug Load (`mels_S`)** | 0.382 | **0.822** | **0.820** | 1.000 | 0.259 | **0.550** | 0.256 | 0.248 |
| **HVAC (`hvac_S`)** | 0.092 | 0.200 | 0.126 | 0.259 | 1.000 | **0.945** | **0.691** | 0.300 |
| **Total South Power** | 0.228 | **0.442** | **0.426** | **0.550** | **0.945** | 1.000 | **0.674** | 0.357 |

### Quantitative Power Difference by Occupancy State

| Occupancy State | Lighting Power (`lig_S`) | Plug Load Power (`mels_S`) | HVAC Power (`hvac_S`) | Total South Wing Power |
|---|---|---|---|---|
| **Unoccupied (`is_occupied = 0`)** | 0.29 kW | 1.96 kW | 17.93 kW | **20.18 kW** |
| **Occupied (`is_occupied = 1`)** | 2.34 kW (**+707%**) | 4.13 kW (**+111%**) | 20.20 kW (**+13%**) | **26.68 kW (+32.2%)** |

**Interpretation:**
- Miscellaneous electric plug loads exhibit an exceptionally strong correlation with human headcount (**r = 0.822**).
- Lighting power correlates strongly with headcount (**r = 0.682**), leaping from a standby baseline of 0.29 kW to 2.34 kW when occupied.
- HVAC power is predominantly driven by weather thermodynamics (**r = 0.691** with outdoor dry-bulb temperature) with a 17.93 kW baseload maintained even when unoccupied.
- This creates an ideal, academically grounded target for the **Optimization Layer**: actively identifying unoccupied states where lighting can be dropped to 0.29 kW or HVAC can enter setback modes.

---

## 7. Data Quality & Cleaning Rules

1. **Sensor Disconnection & Power Reset Outliers:**
   - In `zone_temp_interior.csv`, values of `85.0°C` and `0.0°C` must be replaced with `NaN` and interpolated linearly, as they represent the well-known DS18B20 digital sensor hardware power-on error codes.
2. **Negative Power Drift:**
   - Slight negative readings in `mels_S` and `lig_S` (-0.01 to -0.44 kW) are zero-point calibration drifts of current transducers under zero current. They must be clamped (`np.clip(val, a_min=0, a_max=None)`).
3. **Duplicate Timestamps:**
   - A small number of duplicate timestamps exist in raw high-frequency files (`rtu_*.csv` ~34 to 51 rows; `uft_hw_valve.csv` ~594k rows due to redundant multi-sensor event triggers). 
   - Rule: Group by timestamp and take the mean before resampling.
4. **Time Gaps:**
   - Small gaps (<2 hours) can be safely forward-filled or linearly interpolated.
   - Large missing blocks in non-essential files (e.g., `Unnamed: 6` in `ele.csv` before 2020) should be dropped rather than imputed.

---

## 8. Proposed Alignment, Resampling & Join Strategy

### Spatial Scope: South Wing Focus
Because ground-truth camera occupancy counters (`occ_third_south`, `occ_fourth_south`) were installed specifically in the **South Wing** of Floors 3 & 4, and the electrical submeters cleanly separate South Wing power (`mels_S`, `lig_S`, `hvac_S`), the primary modeling boundary is established as the **South Wing Office Zone**.

### Temporal Alignment: 1-Hour (Hourly) Resampling
While 15-minute modeling is feasible, an **hourly (1H / 60min) modeling resolution** is recommended for the following academic and engineering reasons:
1. **Harmonization of Multi-Rate Sensors:** It seamlessly reconciles 1-minute camera counts, 1-minute RTU controls, 10-minute Raspberry Pi temperatures, and 15-minute electricity/weather records without artificial interpolation distortion.
2. **Noise Reduction:** High-frequency 1-minute noise (e.g. occupants walking to the restroom or momentary door opens) is averaged into a stable occupant-hours signal.
3. **Direct Physical Energy Units:** In an hourly framework, average active power (kW) multiplied by 1 hour equals energy consumption in **kilowatt-hours (kWh)**:
   $$\text{Energy (kWh)} = \overline{P}_{\text{kW}} \times 1\text{ h}$$
4. **BEMS Standard Compliance:** Building Energy Management Systems (ASHRAE Guideline 14 and IPMVP) operate baseline prediction and supervisory control on an hourly basis.

### Mathematical Aggregation Rules:
- **Occupancy:**
  - `occ_total_mean` = $\text{mean}(\text{occ\_third\_south} + \text{occ\_fourth\_south})$ over 60 min.
  - `is_occupied` = $1$ if $\max(\text{occupant count}) > 0$ else $0$.
- **Electrical Power:**
  - `lig_S_kwh`, `mels_S_kwh`, `hvac_S_kwh` = $\text{mean}(P_{\text{kW}})$ over the hour $\times 1\text{ h}$.
  - `south_wing_total_kwh` = $\text{sum}(\text{submeters})$.
- **Environmental & Meteorological:**
  - `outdoor_temp`, `indoor_temp`, `solar_radiation`, `relative_humidity` = $\text{mean}$ over 60 min.
- **HVAC Operations:**
  - `rtu_fan_spd_mean`, `rtu_damper_mean`, `rtu_sa_temp_mean` = $\text{mean}$ of RTU 003 & 004 (serving South Wing).

---

## 9. Machine Learning Target Formulation

### 9.1 Occupancy Prediction (Classification & Regression)
- **Primary Target (Classification):**
  $$\text{is\_occupied} \in \{0, 1\}$$
  Predict whether the zone is occupied during the upcoming hour.
  - Evaluation Metrics: F1-score, Precision, Recall, ROC-AUC, Confusion Matrix. (Class balance: ~56% occupied, 44% unoccupied; well-balanced, avoiding pathological imbalance).
- **Secondary Target (Regression):**
  $$\text{occ\_total\_mean} \in [0, \infty)$$
  Predict the expected number of occupants for fine-grained HVAC ventilation demand calculations.
  - Evaluation Metrics: MAE, RMSE, $R^2$.

### 9.2 Energy Consumption Prediction (Regression)
- **Primary Target (Total Energy):**
  $$\text{south\_wing\_total\_kwh} \in [0, \infty)$$
  Total South Wing electricity consumption (kWh) over the hour.
- **Multi-End-Use Submeter Targets:**
  - `lig_S_kwh`: Lighting energy consumption (kWh)
  - `mels_S_kwh`: Plug-load energy consumption (kWh)
  - `hvac_S_kwh`: HVAC electrical consumption (kWh)
- **Evaluation Metrics:** MAE, RMSE, $R^2$, MAPE.

---

## 10. Data Leakage Prevention Protocol

To ensure strict academic integrity and defensibility:
1. **Preserve Temporal Ordering:** Data points must NEVER be shuffled randomly. K-Fold cross-validation with random shuffling is strictly prohibited.
2. **Chronological Splitting:**
   - **Training Set (70%):** May 22, 2018 – November 30, 2018 (~6.3 months)
   - **Validation Set (15%):** December 1, 2018 – January 10, 2019 (~1.3 months)
   - **Test Set (15%):** January 11, 2019 – February 21, 2019 (~1.4 months)
3. **Strict Lag/Rolling Feature Causality:**
   - Lag features ($y_{t-1}, y_{t-24}$) must only use observations strictly before timestamp $t$.
   - Rolling features must use closed='left' or shift(1) to exclude the current and future values:
     $$\text{rolling\_mean}_{t} = \frac{1}{k} \sum_{i=1}^{k} x_{t-i}$$
4. **Leakage-Free Preprocessing:**
   - Scalers (StandardScaler, MinMaxScaler) and any categorical encoders must be fitted **strictly on the training period** and only applied (transform) to validation and test periods.
5. **No Target Leaks:**
   - Current submeter readings (`lig_S`, `mels_S`) cannot be used to predict concurrent total energy if total energy is their sum.

---

## 11. Proposed Next Phase

With Step 0 complete, the dataset schema, physical correlations, and boundaries are rigorously proven. 

**Recommended Phase 1 & 2 Execution:**
1. Create a lightweight, reproducible data inspection notebook (`notebooks/00_dataset_inspection.ipynb`) storing key verification figures.
2. Implement `src/preprocessing.py` to build the automated, leakage-free pipeline that cleans raw telemetry, aligns the South Wing data, and outputs the processed modeling tables to `data/processed/`.
3. Proceed to **Phase 2: Exploratory Data Analysis (EDA)**.
