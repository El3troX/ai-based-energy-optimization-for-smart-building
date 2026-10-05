# AI-Based Energy Optimization for Smart Buildings

An end-to-end Machine Learning and Optimization system functioning as an intelligent Building Energy Management System (BEMS).

---

## 🏛️ Project Overview

Buildings consume significant electrical energy through HVAC systems, lighting, and plug-loads. Much of this energy is wasted due to static, schedule-based control strategies that do not adapt dynamically to occupancy patterns and real-time environmental conditions.

This project delivers a complete pipeline:
```text
Building Data (Sensors & Meters)
             ↓
      Data Preprocessing
             ↓
     Feature Engineering
             ↓
   +---------+---------+
   |                   |
   v                   v
Occupancy Model   Energy Model
(Classification)  (Regression)
   |                   |
   +---------+---------+
             ↓
    Optimization Engine
 (Constrained Recommendation)
             ↓
   Before/After Simulation
 (Energy / Cost / CO2 Savings)
             ↓
   Interactive Dashboard
        (Streamlit)
```

---

## 📊 Dataset

- **Primary Dataset:** *Three-Year Building Operational Performance Dataset — Building 59* (Lawrence Berkeley National Laboratory)
- **Source:** Dryad repository ([DOI: 10.7941/D1N33Q](https://doi.org/10.7941/D1N33Q))
- **Study Reference:** Luo, Na et al. (2022). *A three-year dataset supporting research on building energy management and occupancy analytics*, Scientific Data.
- **Coverage:** 3 years of continuous operational readings across two office floors (each 2,325 m²) including whole-building electricity, end-use submeters, HVAC system status, indoor climate (temperature, humidity, CO2), outdoor weather, and occupant counts.

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
