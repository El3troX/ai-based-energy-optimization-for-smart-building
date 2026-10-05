"""
Centralized Configuration for Phase 4 BEMS Counterfactual Optimization & Simulation.
Defines:
- Economic and environmental tariffs (PG&E commercial rate, eGRID emissions)
- Empirical candidate control boundaries and ramp constraints
- Occupancy classification vs control-safety thresholds
- Comfort and operational constraint parameters
"""
from typing import Dict, List

# --- Economic and Emissions Parameters ---
# Source: California Commercial Average / PG&E Commercial Rate Schedule B-19
# Unit: USD ($) per kWh
ELECTRICITY_TARIFF: float = 0.22

# Source: EPA eGRID Subregion CAMX (WECC California)
# Unit: kg CO2 equivalent per kWh (approx 0.463 lbs CO2/kWh)
EMISSION_FACTOR: float = 0.210

# --- Historical Support Control Bounds ---
# Derived from empirical training support in Phase 3.5 audit
CONTROL_BOUNDS: Dict[str, Dict[str, float]] = {
    'rtu_south_fan_spd_mean_next_hour': {
        'min': 40.0,
        'max': 90.0,
        'desc': 'RTU South Supply Fan VFD Speed (%)'
    },
    'rtu_south_damper_pct_mean_next_hour': {
        'min': 10.0,
        'max': 90.0,
        'desc': 'RTU South Outdoor Air Damper Position (%)'
    }
}

# --- Operational Ramp Rate Limits ---
# Inter-hour mechanical rate of change limits to prevent equipment cycling and motor wear
RAMP_LIMITS: Dict[str, float] = {
    'fan_speed_max_delta': 15.0,   # Max +/- 15 percentage points per hour
    'damper_pct_max_delta': 30.0    # Max +/- 30 percentage points per hour
}

# --- Occupancy Decision Thresholds ---
# Dual-threshold architecture established in Phase 3 & 3.5
T_CLASS: float = 0.51    # F1-optimal threshold for scientific reporting
T_SAFETY: float = 0.30   # Conservative control-safety gate (FNR <= 1.5% on validation)

# --- Surrogate & Objective Parameters ---
# Minimum estimated energy saving required to recommend an optimization action.
# Rationale: 10% fan speed change in Ridge surrogate corresponds to ~0.435 kWh.
# Epsilon = 0.20 kWh (~4.6% fan adjustment or ~40% damper shift) filters out
# numerical noise and trivial micro-actuations.
MIN_IMPROVEMENT_EPSILON: float = 0.20  # kWh

# Physical baseload floor for clamped display estimates
BASELOAD_ENERGY_FLOOR: float = 6.50  # kWh

# South Wing measured lighting standby power during unoccupied hours
LIGHTING_STANDBY_KWH: float = 0.29  # kWh

# --- Candidate Search Grid ---
SEARCH_GRIDS: Dict[str, List[float]] = {
    'fan_speed': [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0],
    'damper_pct': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
}

# --- Comfort & Operating Mode Rules ---
COMFORT_RULES = {
    'occupied': {
        'min_fan_speed': 50.0,     # ASHRAE 62.1 minimum ventilation airflow
        'min_damper_pct': 20.0,    # Outdoor air ventilation requirement
        'max_fan_speed': 90.0
    },
    'unoccupied': {
        'min_fan_speed': 40.0,     # Low setback ventilation
        'min_damper_pct': 10.0,    # Minimum building pressurization
        'max_fan_speed': 60.0      # Reject high fan when space is unoccupied
    }
}
