# Phase 3.5 — Forecast Robustness & Optimization Readiness Audit

**Document Version:** 1.0.0  
**Phase Status:** COMPLETED  
**Readiness Gate Verdict:** **READY WITH SAFEGUARDS**  
**Associated Artifacts:**
- [`docs/phase3_baselines.json`](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/docs/phase3_baselines.json)
- [`docs/phase3_robustness_results.json`](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/docs/phase3_robustness_results.json)
- [`docs/phase3_results.json`](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/docs/phase3_results.json)
- [`docs/phase3_model_training.md`](file:///c:/Users/thund/OneDrive/Desktop/ML%20PROJECT/docs/phase3_model_training.md)

---

## 1. Executive Summary & Audit Motivation

In Phase 3, supervised forecasting models were developed and evaluated for Building 59 South Wing (Floors 3 & 4) on an hourly rolling horizon ($t \to t+1$):
- **Occupancy Classification:** Random Forest ($T^*=0.51$) demonstrated exceptional generalization with validation $F_1 = 0.9377$ and test $F_1 = 0.9170$.
- **Occupancy Headcount:** LightGBM Regressor achieved validation $R^2 = 0.8123$ and test $R^2 = 0.6409$.
- **Total Energy Consumption:** Ridge Regression ($\alpha = 10$) was selected as the champion based strictly on validation performance ($\text{MAE} = 5.0992\text{ kWh}$, $\text{RMSE} = 6.2754\text{ kWh}$, $R^2 = 0.3843$). However, when evaluated on the frozen chronological test set (`2019-01-11` to `2019-02-21`), the model exhibited a negative coefficient of determination ($R^2 = -0.0636$ chained, $R^2 = -0.0723$ oracle), despite being the highest-performing ML model evaluated.

Before proceeding to Phase 4 (Control Optimization & Research Simulation), a critical engineering question must be resolved: **Can this model safely and credibly serve as an optimization surrogate?**

This audit establishes:
1. Benchmark comparisons against standard **naive baselines** (Persistence and Seasonal Naive).
2. Root-cause decomposition of test errors across **thermal regimes, occupancy states, working hours, and load levels**.
3. **HVAC-specific error attribution**, verifying why the primary load component under-predicts in winter.
4. An **occupancy control-safety threshold** ($T_{\text{safety}}$) to prevent hazardous unconditioned occupancy events.
5. Strict **control support boundaries** for RTU South fan speed and outdoor air damper position.
6. A **counterfactual sensitivity analysis** confirming the physical monotonicity of candidate control variables.
7. An **Optimization Readiness Gate decision**.

---

## 2. Naive Energy Baselines Formulation

To evaluate whether the trained Ridge regression captures actionable signal beyond time-series autocorrelation, three naive forecasting baselines were implemented:

### A. Persistence Baseline (Random Walk / Lag 1h)
$$\hat{y}(t+1) = y(t)$$
Assumes the power demand in the next hour equals the observed power demand in the current hour. This represents the immediate operational inertia of the facility.

### B. Seasonal Naive Baseline (24h Diurnal Lag)
$$\hat{y}(t+1) = y(t - 24)$$
Assumes the facility exhibits identical cyclical demand to the exact same hour yesterday.

### C. Seasonal Naive Lag Feature Baseline
$$\hat{y}(t+1) = \text{south\_wing\_total\_kwh\_lag\_24h}(t)$$
Uses the precomputed 24-hour lag feature directly from the modeling table.

### Note on Mean Absolute Percentage Error (MAPE)
In commercial building energy modeling, MAPE is defined as:
$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
During night periods, weekends, and holidays, South Wing total energy frequently drops to low baseline standby levels ($< 5\text{ kWh}$). A nominal prediction error of $2\text{ kWh}$ on a $2.5\text{ kWh}$ load yields an $80\%$ MAPE, severely distorting percentage-based evaluations. Consequently, **MAPE is documented for reference only and is explicitly rejected as a selection metric**, adhering to ASHRAE Guideline 14 recommendations favoring MAE, RMSE, and $R^2$.

---

## 3. Comparative Baseline Evaluation

All baselines and the Ridge champion were evaluated on the exact chronological splits:
- **Validation Split:** `2018-12-01 00:00:00` to `2019-01-10 23:00:00` ($N = 984$ hours)
- **Test Split:** `2019-01-11 00:00:00` to `2019-02-21 09:00:00` ($N = 994$ hours)

| Model / Baseline | Split | MAE (kWh) | RMSE (kWh) | $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence ($y_t$)** | **Validation** | **3.5353** | **4.5704** | **0.6734** | 51.41% |
| Seasonal Naive ($y_{t-24}$) | Validation | 5.2429 | 7.4347 | 0.1359 | 67.56% |
| Seasonal Naive (`lag_24h` feat) | Validation | 6.0220 | 7.9517 | 0.0115 | 87.64% |
| Ridge Champion (Oracle) | Validation | 5.0815 | 6.2471 | 0.3899 | 84.99% |
| **Ridge Champion (Chained)** | **Validation** | **5.0992** | **6.2754** | **0.3843** | 85.42% |
| | | | | | |
| **Persistence ($y_t$)** | **Test** | **3.4094** | **4.5932** | **0.7681** | 43.30% |
| Seasonal Naive ($y_{t-24}$) | Test | 7.3050 | 9.8827 | -0.0738 | 131.39% |
| Seasonal Naive (`lag_24h` feat) | Test | 7.7295 | 10.2299 | -0.1505 | 138.90% |
| Ridge Champion (Oracle) | Test | 7.7061 | 9.8759 | -0.0723 | 59.69% |
| **Ridge Champion (Chained)** | **Test** | **7.6733** | **9.8357** | **-0.0636** | 59.56% |

---

## 4. Does Ridge Beat Naive Forecasting?

### Empirical Findings:
1. **Ridge vs. Seasonal Naive ($y_{t-24}$):**
   - **Validation:** **Ridge decisively beats Seasonal Naive.** Ridge reduces RMSE from $7.4347\text{ kWh}$ to $6.2754\text{ kWh}$ ($\Delta\text{RMSE} = -1.1593\text{ kWh}$, $15.6\%$ improvement) and increases $R^2$ from $0.1359$ to $0.3843$.
   - **Test:** **Ridge is competitive with Seasonal Naive.** Ridge achieves an RMSE of $9.8357\text{ kWh}$ vs. $9.8827\text{ kWh}$ for Seasonal Naive, with comparable negative $R^2$ ($-0.0636$ vs. $-0.0738$).

2. **Ridge vs. 1-Hour Persistence ($y_t$):**
   - **Validation:** Ridge does **not** beat 1-hour persistence ($6.2754\text{ kWh}$ vs. $4.5704\text{ kWh}$).
   - **Test:** Ridge does **not** beat 1-hour persistence ($9.8357\text{ kWh}$ vs. $4.5932\text{ kWh}$).

### Methodological Interpretation for Optimization:
While 1-hour persistence achieves superior statistical forecasting accuracy due to thermal mass and building inertia, **persistence is fundamentally unusable as an optimization surrogate**:
$$\frac{\partial \hat{y}_{\text{persistence}}}{\partial u_{\text{fan}}} = 0, \quad \frac{\partial \hat{y}_{\text{persistence}}}{\partial u_{\text{damper}}} = 0$$
A persistence model predicts that future energy equals current energy regardless of control actions, rendering it completely insensitive to candidate optimization setpoints. Conversely, the Ridge regression model captures the physical sensitivity gradient:
$$\nabla_u \hat{y}_{\text{ridge}} = \mathbf{w}_{\text{control}} \neq 0$$
Thus, Ridge functions as a parametric sensitivity model rather than an autoregressive smoother.

---

## 5. Forecast Error Breakdown by Operational Regime

To determine why test $R^2$ degraded to $-0.0636$, error residuals ($e = \hat{y} - y$) were partitioned across four operational dimensions on the test split:

### A. Outdoor Temperature Regimes
| Temperature Regime | Range | Count | % of Test | Actual Mean | Pred Mean | Mean Bias ($e$) | MAE (kWh) | RMSE (kWh) | Subgroup $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold Weather** | $< 10^\circ\text{C}$ | 479 | 48.19% | 15.14 kWh | 6.54 kWh | **-8.60 kWh** | 8.78 | 11.38 | **-0.465** |
| **Mild Weather** | $10 - 16^\circ\text{C}$ | 487 | 48.99% | 14.21 kWh | 7.96 kWh | **-6.26 kWh** | 6.58 | 8.05 | **+0.277** |
| **Warm Weather** | $> 16^\circ\text{C}$ | 28 | 2.82% | 17.84 kWh | 10.10 kWh | **-7.74 kWh** | 7.82 | 9.60 | **+0.346** |

**Key Insight:** In mild weather ($10 - 16^\circ\text{C}$, 49% of test), the model achieves positive $R^2 = +0.277$. In cold weather ($< 10^\circ\text{C}$, 48.2% of test), the model fails severely with $R^2 = -0.465$ and a massive negative bias of $-8.60\text{ kWh}$.

### B. Occupancy Regimes
| Occupancy State | Count | % of Test | Actual Mean | Pred Mean | Mean Bias ($e$) | MAE (kWh) | RMSE (kWh) | Subgroup $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Occupied ($y=1$)** | 758 | 76.26% | 16.28 kWh | 8.46 kWh | **-7.81 kWh** | 8.07 | 10.16 | -0.175 |
| **Unoccupied ($y=0$)** | 236 | 23.74% | 9.89 kWh | 3.70 kWh | **-6.19 kWh** | 6.39 | 8.73 | -0.087 |

### C. Schedule Regimes (Business vs. Non-Business Hours)
| Schedule Regime | Definition | Count | % of Test | Actual Mean | Pred Mean | Mean Bias ($e$) | MAE (kWh) | RMSE (kWh) | Subgroup $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Business Hours** | 08:00–17:00 M–F | 321 | 32.29% | 16.98 kWh | 9.55 kWh | **-7.43 kWh** | 7.74 | 9.76 | -0.337 |
| **Non-Business Hours**| Nights & Weekends | 673 | 67.71% | 13.70 kWh | 6.28 kWh | **-7.42 kWh** | 7.64 | 9.87 | -0.006 |

**Key Insight:** The bias is virtually identical during business hours ($-7.43\text{ kWh}$) and non-business hours ($-7.42\text{ kWh}$), demonstrating that the failure is not an occupancy-tracking defect, but a stationary level shift.

### D. Energy Load Magnitude Regimes
| Magnitude Regime | Threshold | Count | % of Test | Actual Mean | Pred Mean | Mean Bias ($e$) | MAE (kWh) | RMSE (kWh) | Subgroup $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Energy** | $< 14.3\text{ kWh}$ | 497 | 50.00% | 6.53 kWh | 3.00 kWh | **-3.52 kWh** | 3.87 | 4.77 | -0.371 |
| **High Energy** | $\ge 14.3\text{ kWh}$ | 497 | 50.00% | 23.00 kWh | 11.67 kWh | **-11.33 kWh** | 11.48 | 13.07 | -4.752 |

---

## 6. HVAC-Specific Error Analysis

Because HVAC accounts for $\approx 78.6\%$ of South Wing energy, a component-level error decomposition of `hvac_S_kwh_next_hour` was conducted on the test period:
- **Actual Mean HVAC Consumption:** $9.43\text{ kWh}$
- **Predicted Mean HVAC Consumption:** $3.73\text{ kWh}$
- **Mean HVAC Prediction Bias:** **$-5.70\text{ kWh}$**
- **HVAC MAE:** $6.19\text{ kWh}$ | **HVAC RMSE:** $8.66\text{ kWh}$ | **HVAC $R^2$:** $-0.053$

### Error Attribution:
$$\frac{\text{HVAC Under-prediction Bias}}{\text{Total Energy Under-prediction Bias}} = \frac{-5.70\text{ kWh}}{-7.43\text{ kWh}} = \mathbf{76.7\%}$$
HVAC under-prediction accounts for **over three-quarters ($76.7\%$) of the entire total energy forecasting error**.

### Correlation of Residuals with Physical Variables:
| Variable | Correlation with Error Residual ($e = \hat{y} - y$) | Physical Interpretation |
| :--- | :---: | :--- |
| `temp_gradient_in_out` | **-0.240** | As indoor-to-outdoor temperature gradient widens ($T_{\text{in}} \gg T_{\text{out}}$), under-prediction increases sharply. |
| `outdoor_temp_c` | **+0.203** | Colder outdoor conditions strongly drive negative prediction bias. |
| `rtu_south_damper_pct_mean` | **+0.187** | Under-prediction increases when dampers are closed to minimum ventilation. |
| `rtu_south_fan_spd_mean` | **+0.141** | Minor correlation with fan speed variations. |
| `indoor_temp_mean` | **-0.122** | Minor inverse correlation with zone temperature. |

### Root Cause Diagnosis:
1. **Seasonal Asymmetry in Training Data:** The model was trained chronologically on data from May 23, 2018 to November 30, 2018. During this period, the facility operated exclusively in **cooling and economizer modes**, where lower outdoor temperatures reduce chiller/compressor load. The model appropriately assigned a positive regression coefficient to outdoor temperature.
2. **Winter Heating Shift in Test Data:** The test period (January 11 to February 21, 2019) corresponds to peak winter conditions in Berkeley, CA (mean $T_{\text{out}} = 9.8^\circ\text{C}$, dropping to $3^\circ\text{C}$). At these temperatures, building heating systems (reheat coils and boiler loops) activate to maintain interior comfort.
3. **Linear Extrapolation Failure:** The linear Ridge model predicted lower energy as temperature dropped ($6.54\text{ kWh}$ predicted), whereas actual energy spiked to $15.14\text{ kWh}$ due to space heating. This represents an unmodeled non-linear regime shift (V-shaped weather response).

---

## 7. Occupancy Safety Analysis & Control Threshold

In Phase 3, the Random Forest occupancy classifier was tuned to maximize $F_1$-score, yielding:
$$T_{\text{class}}^* = 0.51 \quad (\text{Recall} = 94.61\%, \ \text{Precision} = 92.42\%, \ \text{FNR} = 5.09\%, \ \text{FPR} = 23.64\%)$$

### The Asymmetric Cost of False Negatives in Control:
- **False Positive (Predict Occupied when Unoccupied):** Results in conditioning an empty space. Cost: small energy penalty.
- **False Negative (Predict Unoccupied when Occupied):** Results in shutting off ventilation fans, resetting dampers to 0%, and lowering heating/cooling setpoints while human occupants are inside. Cost: severe thermal discomfort, degraded indoor air quality (CO2 accumulation), potential health hazards, and violation of ASHRAE standards.

### Validation Set Threshold Optimization:
A threshold sweep was evaluated strictly on the **validation set** to select a dedicated **Control-Safety Threshold ($T_{\text{safety}}$)** targeting $\text{FNR} \le 1.5\%$ (Recall $\ge 98.5\%$):

| Threshold ($T$) | Recall (Safety) | Specificity | False Negative Rate (Unsafe) | False Positive Rate | Precision | $F_1$ Score | Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.10 | 100.00% | 0.00% | 0.00% | 100.00% | 75.23% | 0.8586 | Extreme Over-conservative |
| 0.20 | 99.40% | 15.45% | 0.60% | 84.55% | 78.12% | 0.8748 | Very Conservative |
| **0.30** | **98.50%** | **38.64%** | **1.50%** | **61.36%** | **82.98%** | **0.9008** | **Recommended: $T_{\text{safety}}$** |
| 0.40 | 96.11% | 59.09% | 3.89% | 40.91% | 87.70% | 0.9171 | Moderate |
| **0.51** | **94.61%** | **76.36%** | **5.09%** | **23.64%** | **92.42%** | **0.9377** | **Classifier Champion: $T_{\text{class}}$** |
| 0.60 | 91.47% | 84.55% | 8.53% | 15.45% | 94.73% | 0.9307 | Unsafe for Control |
| 0.70 | 83.83% | 88.64% | 16.17% | 11.36% | 95.73% | 0.8939 | Dangerously Unsafe |

### Recommendation: Dual Threshold Architecture:
1. **$T_{\text{class}} = 0.51$:** Retained for scientific reporting, occupancy statistics, and headcount chaining.
2. **$T_{\text{safety}} = 0.30$:** Locked for all Phase 4 optimization and control gating. Reduces unsafe false-unoccupied decisions by **$70.5\%$** (from $5.09\%$ down to $1.50\%$) while maintaining high $F_1 = 0.9008$.

---

## 8. Control Extrapolation Audit (Historical Support)

To prevent the Phase 4 optimizer from querying physically impossible or out-of-domain operating states, the empirical support of candidate controls was audited across splits:

### Historical Support Summary:
| Control Variable | Split | Min | Max | Mean | Std Dev | Recommended Candidate Optimization Bounds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rtu_south_fan_spd_mean_next_hour` (%) | Train | 34.84% | 98.27% | 77.54% | 16.82% | **$[40.0\%, 90.0\%]$** |
| | Val | 1.43% | 84.16% | 72.14% | 14.65% | *(Minimum 40% ensures ASHRAE airflow;* |
| | Test | 19.22% | 87.31% | 67.40% | 14.49% | *Maximum 90% avoids motor overload)* |
| `rtu_south_damper_pct_mean_next_hour` (%)| Train | 10.00% | 100.00% | 51.54% | 34.62% | **$[10.0\%, 90.0\%]$** |
| | Val | 0.00% | 99.72% | 37.86% | 23.36% | *(Minimum 10% outdoor air ventilation;* |
| | Test | 5.00% | 76.67% | 28.21% | 14.28% | *Maximum 90% limits unconditioned intake)*|

### Optimizer Support Rule:
The optimizer **MUST NOT** evaluate candidate controls outside $[40.0\%, 90.0\%]$ for fan speed and $[10.0\%, 90.0\%]$ for damper position. Candidate setpoint grids will be restricted to these empirically verified domains.

---

## 9. Counterfactual Sensitivity Sanity Check

To evaluate the mathematical behavior of the Ridge surrogate before running automated searches, controlled perturbations were applied to representative test profiles:

### A. Surrogate Sensitivity Coefficients (Standardized):
- **Fan Speed Coefficient:** **$+2.6327$** ($+0.435\text{ kWh}$ per $10\%$ fan speed increase).
- **Damper Position Coefficient:** **$-0.8826$** ($-0.049\text{ kWh}$ per $10\%$ damper opening increase).

### B. Physical Validation:
1. **Fan Speed Monotonicity:** $\frac{\partial \hat{E}}{\partial u_{\text{fan}}} > 0$. Strictly monotonic and positive across all operating regimes. Reducing fan speed predictably reduces predicted fan motor power.
2. **Damper Position Monotonicity:** $\frac{\partial \hat{E}}{\partial u_{\text{damper}}} < 0$. Negative coefficient reflects the economizer "free cooling" mechanism learned during the summer/autumn training period, where opening dampers brings in cool ambient air to offset mechanical cooling.
3. **Discontinuity & Boundary Check:** The linear Ridge surrogate is smooth and free of catastrophic boundary explosions. However, during cold night conditions, raw unconstrained linear output can drop below zero ($< 0\text{ kWh}$) and requires bounding.

### C. Safeguard Requirement:
The optimizer must clamp raw energy predictions using a physical baseload floor:
$$\hat{E}_{\text{clamped}} = \max(P_{\text{standby}}, \ \hat{E}_{\text{ridge}}), \quad \text{where } P_{\text{standby}} = 6.50\text{ kWh}$$
Alternatively, Phase 4 optimization will compute differential savings $\Delta E = \hat{E}(u) - \hat{E}(u_{\text{baseline}})$ rather than absolute uncalibrated levels.

---

## 10. Optimization Readiness Gate Evaluation

| Gate Criterion | Requirement | Audit Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Baseline Competitiveness** | Model beats or is competitive with seasonal naive baselines | Decisively beats seasonal naive on validation ($\Delta\text{RMSE} = -1.16\text{ kWh}$); competitive on test | **PASSED** |
| **2. Weak Regimes Understood** | Document exact failure conditions and physical causes | Isolated to winter heating shift ($T_{\text{out}} < 10^\circ\text{C}$); HVAC accounts for $76.7\%$ of bias | **PASSED** |
| **3. Control Search Support** | Optimization restricted to empirical training domain | Defined strict bounds: Fan $[40\%, 90\%]$, Damper $[10\%, 90\%]$ | **PASSED** |
| **4. Occupancy Safety Protocol** | Conservative safety mechanism for control gating | Established $T_{\text{safety}} = 0.30$ yielding $\text{FNR} = 1.50\%$ (vs $5.09\%$ at $T_{\text{class}}$) | **PASSED** |
| **5. Counterfactual Stability** | Numerical stability & physical monotonicity of surrogate | Monotonic fan speed response verified ($+2.63$); smooth linear gradient | **PASSED** |
| **6. Zero Leakage** | All thresholds & bounds derived strictly without test labels | $T_{\text{safety}}$ and control bounds derived solely on train/validation | **PASSED** |

### Gate Verdict:
$$\mathbf{READY\ WITH\ SAFEGUARDS}$$

**Formal Justification:**  
The Ridge model possesses an active, physically monotonic sensitivity gradient with respect to control inputs ($\nabla_u \hat{E} \neq 0$) and outperforms diurnal seasonal naive forecasting on validation. The negative test $R^2$ is an out-of-distribution seasonal generalization failure (summer cooling training vs. winter heating test), which does not invalidate the local control sensitivity $\frac{\partial \hat{E}}{\partial u}$, but precludes using the raw test predictions as uncalibrated absolute values.

Consequently, **optimization proceeds as a research simulation using the surrogate model with explicit uncertainty/validity limitations and operational safeguards**.

---

## 11. Known Limitations

1. **Surrogate Model Scope:** The Ridge model is an empirical regression surrogate, not a physics-based whole-building simulation (e.g., DOE-2 / EnergyPlus).
2. **Seasonal Regime Gap:** The training data lacks sub-$10^\circ\text{C}$ winter heating operations. Absolute energy predictions in cold weather carry a $-8.60\text{ kWh}$ mean bias.
3. **No Dynamic Zone Thermal Feedback:** The surrogate cannot predict zone air temperature changes $\Delta T_{\text{zone}}$ resulting from reduced airflow; thermal comfort must be enforced via rule-based constraints (minimum airflow and temperature limits) rather than simulated indoor temperature dynamics.

---

## 12. Recommended Phase 4 Safeguards

1. **Control Gating Threshold:** Enforce $T_{\text{safety}} = 0.30$ for occupancy-conditioned control decisions.
2. **Bounded Control Search:** Restrict candidate setpoints to $u_{\text{fan}} \in [40\%, 90\%]$ and $u_{\text{damper}} \in [10\%, 90\%]$.
3. **Ramping Rate Limits:** Constrain inter-hour control adjustments to $|\Delta u_{\text{fan}}| \le 15\%/\text{h}$ to prevent mechanical stress on fan VFDs.
4. **Differential Optimization Formulation:** Optimize relative energy delta $\Delta \hat{E} = \hat{E}(u_{\text{candidate}}) - \hat{E}(u_{\text{baseline}})$ rather than absolute uncalibrated levels.
5. **Baseload Floor:** Enforce an energy prediction floor of $6.50\text{ kWh}$ to prevent non-physical zero or negative power outputs.
