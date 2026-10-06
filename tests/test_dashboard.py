"""
Automated Unit and Integration Tests for Phase 5 Streamlit BEMS Dashboard.
Verifies data loading, model artifact retrieval, single-hour scenario optimization,
output schemas, safety gating, and graceful error handling without launching a browser.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.config import (
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    BASELOAD_ENERGY_FLOOR
)
from dashboard.utils import (
    load_simulation_results,
    load_joint_test_data,
    get_cached_models,
    load_benchmark_metrics,
    load_robustness_metrics,
    load_phase4_summary,
    run_scenario_optimization,
    get_representative_timestamps
)


def test_dashboard_app_imports_cleanly():
    """Verify dashboard/app.py and dashboard/utils.py can be imported without error."""
    import dashboard.app
    import dashboard.utils
    assert hasattr(dashboard.app, "main")
    assert hasattr(dashboard.utils, "run_scenario_optimization")


def test_required_data_and_metric_files_exist():
    """Verify that all prerequisite processed tables and benchmark JSON files exist."""
    sim_path = Path("data/processed/phase4_simulation_results.parquet")
    joint_path = Path("data/processed/joint_modeling_data.parquet")
    p3_path = Path("docs/phase3_results.json")
    p3_rob_path = Path("docs/phase3_robustness_results.json")
    p4_path = Path("docs/phase4_results.json")
    
    assert sim_path.exists(), f"Missing {sim_path}"
    assert joint_path.exists(), f"Missing {joint_path}"
    assert p3_path.exists(), f"Missing {p3_path}"
    assert p3_rob_path.exists(), f"Missing {p3_rob_path}"
    assert p4_path.exists(), f"Missing {p4_path}"


def test_serialized_models_load_successfully():
    """Verify serialized champion models and scalers load with valid schemas."""
    occ_art, energy_art = get_cached_models()
    
    assert "model" in occ_art
    assert "scaler" in occ_art
    assert "feature_names" in occ_art
    assert len(occ_art["feature_names"]) > 0
    
    assert "model" in energy_art
    assert "scaler" in energy_art
    assert "feature_names" in energy_art
    assert "rtu_south_fan_spd_mean_next_hour" in energy_art["feature_names"]
    assert "rtu_south_damper_pct_mean_next_hour" in energy_art["feature_names"]


def test_simulation_data_schema_and_integrity():
    """Verify the loaded simulation dataset has 994 rows and all expected columns."""
    df = load_simulation_results()
    assert len(df) == 994
    expected_cols = [
        "occupancy_probability",
        "occupancy_state",
        "control_safety_gate",
        "baseline_fan",
        "baseline_damper",
        "recommended_fan",
        "recommended_damper",
        "baseline_predicted_energy",
        "optimized_predicted_energy",
        "optimized_delta_energy",
        "estimated_energy_saving",
        "estimated_saving_percent",
        "estimated_cost_saving",
        "estimated_co2_reduction",
        "action_reason",
        "actual_total_energy",
        "outdoor_temp_c"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
        
    assert (df["occupancy_probability"] >= 0.0).all() and (df["occupancy_probability"] <= 1.0).all()
    assert (df["baseline_predicted_energy"] >= BASELOAD_ENERGY_FLOOR).all()
    assert (df["optimized_predicted_energy"] >= BASELOAD_ENERGY_FLOOR).all()


def test_representative_scenarios_exist_and_execute():
    """Verify all curated representative historical timestamps exist in test set and execute cleanly."""
    joint_df = load_joint_test_data()
    occ_art, energy_art = get_cached_models()
    rep_scenarios = get_representative_timestamps()
    
    assert len(rep_scenarios) >= 5
    for name, ts_str in rep_scenarios.items():
        ts = pd.to_datetime(ts_str)
        assert ts in joint_df.index, f"Representative timestamp {ts_str} not in test index"
        
        res = run_scenario_optimization(
            timestamp=ts,
            joint_df=joint_df,
            occ_art=occ_art,
            energy_art=energy_art,
            enforce_ramps=True,
            enforce_comfort=True
        )
        assert isinstance(res, dict)
        assert "recommended_fan" in res
        assert "recommended_damper" in res
        assert "optimized_delta_energy" in res
        assert res["optimized_delta_energy"] <= 0.0001  # Must not increase energy


def test_scenario_optimization_output_schema():
    """Verify output schema of single-hour scenario optimization."""
    joint_df = load_joint_test_data()
    occ_art, energy_art = get_cached_models()
    sample_ts = joint_df.index[50]
    
    res = run_scenario_optimization(
        timestamp=sample_ts,
        joint_df=joint_df,
        occ_art=occ_art,
        energy_art=energy_art
    )
    
    required_keys = [
        "timestamp", "occupancy_probability", "occupancy_state",
        "control_safety_gate", "baseline_fan", "baseline_damper",
        "recommended_fan", "recommended_damper",
        "baseline_predicted_energy", "optimized_predicted_energy",
        "raw_delta_energy", "optimized_delta_energy",
        "estimated_energy_saving", "estimated_saving_percent",
        "estimated_cost_saving", "estimated_co2_reduction",
        "action_reason", "constraint_status", "confidence_flag"
    ]
    for key in required_keys:
        assert key in res, f"Key {key} missing from optimization result"
        
    assert CONTROL_BOUNDS["rtu_south_fan_spd_mean_next_hour"]["min"] <= res["recommended_fan"] <= CONTROL_BOUNDS["rtu_south_fan_spd_mean_next_hour"]["max"]
    assert CONTROL_BOUNDS["rtu_south_damper_pct_mean_next_hour"]["min"] <= res["recommended_damper"] <= CONTROL_BOUNDS["rtu_south_damper_pct_mean_next_hour"]["max"]


def test_scenario_overrides_functionality():
    """Verify user what-if overrides alter optimization inputs and result."""
    joint_df = load_joint_test_data()
    occ_art, energy_art = get_cached_models()
    sample_ts = joint_df.index[100]
    
    # Run with default
    res_base = run_scenario_optimization(sample_ts, joint_df, occ_art, energy_art)
    
    # Run with fan override to 85%
    overrides = {"rtu_south_fan_spd_mean_next_hour": 85.0}
    res_ov = run_scenario_optimization(sample_ts, joint_df, occ_art, energy_art, overrides=overrides)
    
    assert res_ov["baseline_fan"] == 85.0


def test_graceful_error_handling_on_invalid_timestamp():
    """Verify invalid timestamp raises KeyError cleanly rather than unhandled crash."""
    joint_df = load_joint_test_data()
    occ_art, energy_art = get_cached_models()
    invalid_ts = pd.to_datetime("2025-01-01 00:00:00")
    
    with pytest.raises(KeyError):
        run_scenario_optimization(invalid_ts, joint_df, occ_art, energy_art)


def test_safety_gate_occupancy_protection():
    """Verify that when occupancy probability exceeds T_safety, safety gate status is PROTECTED_OCCUPIED."""
    sim_df = load_simulation_results()
    high_occ_hours = sim_df[sim_df["occupancy_probability"] >= T_SAFETY]
    assert len(high_occ_hours) > 0
    assert (high_occ_hours["control_safety_gate"] == "PROTECTED_OCCUPIED").all()
