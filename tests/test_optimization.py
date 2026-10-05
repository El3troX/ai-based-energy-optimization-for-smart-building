"""
Automated Test Suite for Phase 4 BEMS Optimization and Simulation.
Tests:
1. Control bounds enforcement
2. Occupancy safety gate (T=0.30 vs T=0.51)
3. Fan ramp rate limit (<= 15 percentage points/hr)
4. Candidate generation validity
5. Baseline comparison & zero differential identity
6. NO_CHANGE behavior when improvement < epsilon
7. Energy floor enforcement (>= 6.50 kWh)
8. Invalid input rejection (comfort and operational violations)
9. Candidate feature matrix compatibility
10. Optimization output schema conformity
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON,
    BASELOAD_ENERGY_FLOOR
)
from src.optimizer import (
    load_optimization_models,
    apply_occupancy_safety_gate,
    validate_control_candidate,
    generate_candidate_grid,
    predict_candidate_energy,
    optimize_hour
)
from src.simulator import (
    simulate_period,
    summarize_simulation
)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJ_ROOT / 'data' / 'processed' / 'joint_modeling_data.parquet'
OCC_MODEL_DIR = PROJ_ROOT / 'models' / 'occupancy'
ENERGY_MODEL_DIR = PROJ_ROOT / 'models' / 'energy'


@pytest.fixture(scope="module")
def optimization_models():
    return load_optimization_models(str(OCC_MODEL_DIR), str(ENERGY_MODEL_DIR))


@pytest.fixture(scope="module")
def sample_test_data():
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    test_mask = (df.index >= '2019-01-11 00:00:00') & (df.index <= '2019-02-21 09:00:00')
    return df.loc[test_mask]


def test_control_bounds():
    """Test 1: Verifies that controls outside [40, 90] fan and [10, 90] damper are rejected."""
    # Valid controls
    valid, msg = validate_control_candidate(
        candidate_fan=60.0, candidate_damper=40.0,
        baseline_fan=60.0, baseline_damper=40.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=False
    )
    assert valid is True
    assert msg == "VALID"
    
    # Fan below min (35 < 40)
    valid, msg = validate_control_candidate(
        candidate_fan=35.0, candidate_damper=40.0,
        baseline_fan=60.0, baseline_damper=40.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=False
    )
    assert valid is False
    assert "outside supported range" in msg
    
    # Fan above max (95 > 90)
    valid, msg = validate_control_candidate(
        candidate_fan=95.0, candidate_damper=40.0,
        baseline_fan=60.0, baseline_damper=40.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=False
    )
    assert valid is False
    assert "outside supported range" in msg
    
    # Damper below min (5 < 10)
    valid, msg = validate_control_candidate(
        candidate_fan=60.0, candidate_damper=5.0,
        baseline_fan=60.0, baseline_damper=40.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=False
    )
    assert valid is False
    assert "outside supported range" in msg


def test_occupancy_safety_gate():
    """Test 2: Verifies safety gate behavior at T_safety=0.30 vs classification T=0.51."""
    # Under safety threshold
    assert apply_occupancy_safety_gate(0.15, safety_threshold=T_SAFETY) is False
    assert apply_occupancy_safety_gate(0.29, safety_threshold=T_SAFETY) is False
    
    # Critical ambiguity zone: 0.30 <= prob < 0.51
    # Must be PROTECTED occupied for control even if classified unoccupied
    prob = 0.40
    assert apply_occupancy_safety_gate(prob, safety_threshold=T_SAFETY) is True
    assert (prob >= T_CLASS) is False # Classified 0, but protected 1!
    
    # Above classification threshold
    assert apply_occupancy_safety_gate(0.75, safety_threshold=T_SAFETY) is True


def test_fan_ramp_limit():
    """Test 3: Verifies enforcement of inter-hour ramp limit (max 15%)."""
    baseline_fan = 60.0
    baseline_damper = 40.0
    
    # Within ramp limit: 60 -> 70 (+10)
    valid, msg = validate_control_candidate(
        candidate_fan=70.0, candidate_damper=40.0,
        baseline_fan=baseline_fan, baseline_damper=baseline_damper,
        is_control_occupied=True, enforce_ramps=True, enforce_comfort=False
    )
    assert valid is True
    
    # Violating ramp limit: 60 -> 80 (+20 > 15)
    valid, msg = validate_control_candidate(
        candidate_fan=80.0, candidate_damper=40.0,
        baseline_fan=baseline_fan, baseline_damper=baseline_damper,
        is_control_occupied=True, enforce_ramps=True, enforce_comfort=False
    )
    assert valid is False
    assert "Fan ramp" in msg


def test_candidate_generation():
    """Test 4: Verifies candidate grid generates valid points within support and ramps."""
    baseline_fan = 65.0
    baseline_damper = 50.0
    candidates = generate_candidate_grid(
        baseline_fan=baseline_fan,
        baseline_damper=baseline_damper,
        is_control_occupied=True,
        enforce_ramps=True,
        enforce_comfort=True
    )
    assert len(candidates) > 0
    for fan, damper in candidates:
        assert 40.0 <= fan <= 90.0
        assert 10.0 <= damper <= 90.0
        assert abs(fan - baseline_fan) <= RAMP_LIMITS['fan_speed_max_delta']
        assert abs(damper - baseline_damper) <= RAMP_LIMITS['damper_pct_max_delta']


def test_baseline_comparison(sample_test_data, optimization_models):
    """Test 5: Verifies that predicting energy for baseline controls produces Delta E = 0."""
    occ_art, energy_art = optimization_models
    model = energy_art['model']
    scaler = energy_art['scaler']
    features = energy_art['feature_names']
    
    row = sample_test_data.iloc[10]
    base_dict = {f: row[f] for f in features if f in row}
    fan = float(row['rtu_south_fan_spd_mean_next_hour'])
    damper = float(row['rtu_south_damper_pct_mean_next_hour'])
    
    pred1 = predict_candidate_energy(model, scaler, features, base_dict, fan, damper, 1)
    pred2 = predict_candidate_energy(model, scaler, features, base_dict, fan, damper, 1)
    
    assert abs(pred1 - pred2) < 1e-6


def test_no_change_behavior(sample_test_data, optimization_models):
    """Test 6: Verifies NO_CHANGE returned when improvement does not exceed epsilon."""
    occ_art, energy_art = optimization_models
    row = sample_test_data.iloc[20].copy()
    row.name = sample_test_data.index[20]
    
    # Set an enormous epsilon that no candidate can satisfy
    res = optimize_hour(
        row=row,
        occ_model_artifacts=occ_art,
        energy_model_artifacts=energy_art,
        min_improvement_epsilon=1000.0 # 1000 kWh threshold
    )
    
    assert "No candidate exceeded improvement threshold" in res['action_reason']
    assert res['estimated_energy_saving'] == 0.0
    assert res['recommended_fan'] == res['baseline_fan']
    assert res['recommended_damper'] == res['baseline_damper']


def test_energy_floor(sample_test_data, optimization_models):
    """Test 7: Verifies baseline and optimized display predictions never fall below baseload floor."""
    occ_art, energy_art = optimization_models
    row = sample_test_data.iloc[5].copy()
    row.name = sample_test_data.index[5]
    
    res = optimize_hour(
        row=row,
        occ_model_artifacts=occ_art,
        energy_model_artifacts=energy_art
    )
    
    assert res['baseline_predicted_energy'] >= BASELOAD_ENERGY_FLOOR
    assert res['optimized_predicted_energy'] >= BASELOAD_ENERGY_FLOOR


def test_invalid_input_rejection():
    """Test 8: Verifies rejection of comfort-violating states."""
    # Occupied space with low ventilation fan (40% < min 50%)
    valid, msg = validate_control_candidate(
        candidate_fan=40.0, candidate_damper=30.0,
        baseline_fan=50.0, baseline_damper=30.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=True
    )
    assert valid is False
    assert "below comfort min" in msg
    
    # Occupied space with closed outdoor air damper (10% < min 20%)
    valid, msg = validate_control_candidate(
        candidate_fan=60.0, candidate_damper=10.0,
        baseline_fan=60.0, baseline_damper=20.0,
        is_control_occupied=True, enforce_ramps=False, enforce_comfort=True
    )
    assert valid is False
    assert "below comfort min" in msg


def test_candidate_feature_compatibility(sample_test_data, optimization_models):
    """Test 9: Verifies predict_candidate_energy works across all test rows without schema mismatch."""
    _, energy_art = optimization_models
    model = energy_art['model']
    scaler = energy_art['scaler']
    features = energy_art['feature_names']
    
    for i in [0, 50, 100]:
        row = sample_test_data.iloc[i]
        base_dict = {f: row[f] for f in features if f in row}
        val = predict_candidate_energy(model, scaler, features, base_dict, 60.0, 30.0, 1)
        assert isinstance(val, float)
        assert not np.isnan(val)


def test_optimization_output_schema(sample_test_data, optimization_models):
    """Test 10: Verifies all required keys in optimization output dictionary."""
    occ_art, energy_art = optimization_models
    row = sample_test_data.iloc[15].copy()
    row.name = sample_test_data.index[15]
    
    res = optimize_hour(
        row=row,
        occ_model_artifacts=occ_art,
        energy_model_artifacts=energy_art
    )
    
    required_keys = [
        'timestamp',
        'occupancy_probability',
        'occupancy_state',
        'control_safety_gate',
        'baseline_fan',
        'baseline_damper',
        'recommended_fan',
        'recommended_damper',
        'baseline_predicted_energy',
        'optimized_predicted_energy',
        'raw_delta_energy',
        'optimized_delta_energy',
        'estimated_energy_saving',
        'estimated_saving_percent',
        'estimated_cost_saving',
        'estimated_co2_reduction',
        'action_reason',
        'constraint_status',
        'confidence_flag'
    ]
    
    for key in required_keys:
        assert key in res, f"Missing required key '{key}' in optimization result"
