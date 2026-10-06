"""
BEMS Dashboard Utility Module (Phase 5).
Provides cached data loading, model artifact retrieval, single-hour scenario optimization,
and analytics helper functions for the Streamlit interactive dashboard.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON,
    BASELOAD_ENERGY_FLOOR,
    LIGHTING_STANDBY_KWH
)
from src.optimizer import (
    load_optimization_models,
    optimize_hour,
    generate_candidate_grid,
    compute_surrogate_confidence,
    apply_occupancy_safety_gate
)


@st.cache_data(show_spinner=False)
def load_simulation_results(
    file_path: str = "data/processed/phase4_simulation_results.parquet"
) -> pd.DataFrame:
    """
    Loads precomputed Phase 4 994-hour simulation results.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation results file not found at: {file_path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def load_joint_test_data(
    file_path: str = "data/processed/joint_modeling_data.parquet"
) -> pd.DataFrame:
    """
    Loads joint modeling table filtered to the frozen test period (2019-01-11 to 2019-02-21).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Joint modeling data file not found at: {file_path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    test_mask = (df.index >= "2019-01-11 00:00:00") & (df.index <= "2019-02-21 09:00:00")
    return df.loc[test_mask].copy()


@st.cache_resource(show_spinner=False)
def get_cached_models(
    occ_dir: str = "models/occupancy",
    energy_dir: str = "models/energy"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads and caches serialized Random Forest occupancy and Ridge energy artifacts.
    """
    return load_optimization_models(occ_dir, energy_dir)


@st.cache_data(show_spinner=False)
def load_benchmark_metrics() -> Dict[str, Any]:
    """
    Loads precomputed Phase 3 benchmark metrics from docs/phase3_results.json.
    """
    p = Path("docs/phase3_results.json")
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_robustness_metrics() -> Dict[str, Any]:
    """
    Loads precomputed Phase 3.5 robustness metrics from docs/phase3_robustness_results.json.
    """
    p = Path("docs/phase3_robustness_results.json")
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_phase4_summary() -> Dict[str, Any]:
    """
    Loads precomputed Phase 4 simulation summary from docs/phase4_results.json.
    """
    p = Path("docs/phase4_results.json")
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_scenario_optimization(
    timestamp: pd.Timestamp,
    joint_df: pd.DataFrame,
    occ_art: Dict[str, Any],
    energy_art: Dict[str, Any],
    overrides: Optional[Dict[str, float]] = None,
    enforce_ramps: bool = True,
    enforce_comfort: bool = True,
    include_lighting: bool = True
) -> Dict[str, Any]:
    """
    Retrieves the real historical feature row for the given timestamp, applies optional overrides,
    and runs the optimizer to return baseline and counterfactual metrics.
    """
    if timestamp not in joint_df.index:
        raise KeyError(f"Timestamp {timestamp} not present in test dataset.")
    
    row = joint_df.loc[timestamp].copy()
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
        
    # Apply user overrides if provided
    if overrides:
        for k, v in overrides.items():
            if k in row.index:
                row[k] = v
                
    result = optimize_hour(
        row=row,
        occ_model_artifacts=occ_art,
        energy_model_artifacts=energy_art,
        safety_threshold=T_SAFETY,
        classification_threshold=T_CLASS,
        enforce_ramps=enforce_ramps,
        enforce_comfort=enforce_comfort,
        min_improvement_epsilon=MIN_IMPROVEMENT_EPSILON,
        include_lighting=include_lighting
    )
    
    # Enrich with timestamp and input features for display
    result["timestamp"] = timestamp
    result["indoor_temp_c"] = float(row.get("indoor_temp_mean", np.nan))
    result["outdoor_temp_c"] = float(row.get("outdoor_temp_c", np.nan))
    result["relative_humidity"] = float(row.get("relative_humidity", np.nan))
    result["solar_radiation"] = float(row.get("solar_radiation", np.nan))
    result["actual_total_energy"] = float(row.get("south_wing_total_kwh_next_hour", np.nan))
    result["actual_is_occupied"] = int(row.get("is_occupied_next_hour", 0))
    
    return result


def get_representative_timestamps() -> Dict[str, str]:
    """
    Returns a curated set of representative historical test hours for fast scenario selection.
    """
    return {
        "Cold Winter Night (Unoccupied, High Damper)": "2019-01-14 02:00:00",
        "Peak Business Hours (Occupied, Heavy Load)": "2019-01-15 14:00:00",
        "Morning Warmup / Occupancy Transition": "2019-01-16 08:00:00",
        "Mild Afternoon (Partially Occupied)": "2019-01-23 15:00:00",
        "Weekend Daytime Setback (Unoccupied)": "2019-01-20 12:00:00",
        "Cold Morning Surge (High Delta T)": "2019-02-05 09:00:00",
        "Evening Wind-Down (Occupancy Exit)": "2019-02-12 19:00:00",
        "Surrogate Boundary Condition (Mildest Day)": "2019-02-15 13:00:00"
    }
