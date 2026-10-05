# AI-Based Energy Optimization for Smart Buildings

An end-to-end Machine Learning and Optimization system functioning as an intelligent Building Energy Management System (BEMS).

---

## 🏛️ Project Overview

Buildings consume significant electrical energy through HVAC systems, lighting, and plug-loads. Much of this energy is wasted due to static, schedule-based control strategies that do not adapt dynamically to occupancy patterns and real-time environmental conditions.

This project delivers a complete pipeline:
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

---

## 🚦 Project Status & Roadmap

| Phase | Description | Status | Deliverables |
|---|---|---|---|
| **Phase 0** | Repository Architecture & Environments | **COMPLETED** | Repo, `.gitignore`, `requirements.txt` |
| **Step 0** | Dataset Inspection & Schema Discovery | **COMPLETED** | `docs/dataset.md`, `docs/dataset_inventory.csv` |
| **Phase 1** | Modeling Table Construction ($t \to t+1$) | **COMPLETED** | `data/processed/*.parquet`, `docs/phase1_modeling_tables.md` |
| **Audit** | Final Pre-ML Implementation vs Spec Audit | **PASSED** | `docs/pre_ml_audit.md`, `tests/test_preprocessing.py` (9/9 passing) |
| **Phase 2** | Exploratory Data Analysis (EDA) | **COMPLETED** | `notebooks/01_eda.ipynb`, `docs/phase2_eda_report.md`, `docs/figures/eda/` |
| **Phase 3** | Model Training & Evaluation | **COMPLETED** | `src/occupancy_model.py`, `src/energy_model.py`, `docs/phase3_model_training.md` |
| **Phase 4** | Counterfactual Optimization & Simulation | **NEXT** | `src/optimizer.py`, `src/simulator.py` |
| **Phase 5** | Interactive Streamlit BEMS Dashboard | **PENDING** | `dashboard/app.py` |

---

## 📊 Dataset

- **Primary Dataset:** *Three-Year Building Operational Performance Dataset — Building 59* (Lawrence Berkeley National Laboratory)
- **Source:** Dryad repository ([DOI: 10.7941/D1N33Q](https://doi.org/10.7941/D1N33Q))
- **Study Reference:** Luo, Na et al. (2022). *A three-year dataset supporting research on building energy management and occupancy analytics*, Scientific Data.
- **Coverage:** 275-day verified sensor overlap (May 22, 2018 to February 21, 2019) across South Wing office floors (Floors 3 & 4) including whole-building electricity, end-use submeters, HVAC system status, indoor climate, outdoor weather, and camera occupant counts.

---

## 📁 Repository Structure

```text
ai-based-energy-optimization-for-smart-building/
│
├── data/
│   ├── raw/               # Immutable raw data files
│   └── processed/         # Resampled, cleaned, and merged data
│
├── notebooks/             # Exploratory Data Analysis and experimentation
│
├── src/                   # Production modular pipeline code
│   ├── __init__.py
│   ├── preprocessing.py   # Leakage-free cleaning and transformation
│   ├── features.py        # Temporal, lag, and environmental features
│   ├── occupancy_model.py # Occupancy prediction models
│   ├── energy_model.py    # Energy consumption regression models
│   ├── optimizer.py       # Optimization layer (comfort/energy constraints)
│   ├── simulator.py       # Baseline vs optimized operational simulation
│   └── evaluation.py      # Standardized evaluation metrics
│
├── models/                # Serialized model artifacts and scalers
├── dashboard/             # Interactive Streamlit BEMS dashboard
├── docs/                  # Detailed data dictionary and reports
├── tests/                 # Automated unit and integration tests
├── Context.md             # Project specification & guidelines
├── Project Plan.md        # Step-by-step development roadmap
├── requirements.txt       # Project dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/El3troX/ai-based-energy-optimization-for-smart-building.git
   cd ai-based-energy-optimization-for-smart-building
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔬 Academic Standards & Methodology

- **Time-Series Integrity:** Strict chronological splits (no random shuffling that leaks future observations into past states).
- **Leakage Prevention:** Feature transformations, encoders, and scalers are fitted exclusively on training sets.
- **Empirical Rigor:** All reported performance metrics (MAE, RMSE, R², F1, Accuracy) and calculated savings are derived strictly from reproducible experiments on actual data.
