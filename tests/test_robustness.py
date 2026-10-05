"""
Automated Test Suite for Phase 3.5 Robustness & Optimization Readiness Audit.
Tests:
1. Naive baseline calculation (Persistence and Seasonal Naive)
2. No train/test contamination in baseline and threshold calculations
3. Control range enforcement and clamping
4. Threshold safety calculation (FNR <= 1.5%, monotonicity of recall)
5. Counterfactual input validation & surrogate monotonicity
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.energy_model import load_energy_model_artifacts
from src.occupancy_model import load_occupancy_model_artifacts
from src.robustness import (
    compute_persistence_baseline,
    compute_seasonal_naive_baseline,
    evaluate_naive_baseline,
    compute_occupancy_safety_curve,
    select_control_safety_threshold,
    validate_control_bounds,
    clamp_controls,
    evaluate_counterfactual_response,
    CONTROL_BOUNDS,
    RECOMMENDED_SAFETY_THRESHOLD,
    RECOMMENDED_CLASS_THRESHOLD
)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJ_ROOT / 'data' / 'processed' / 'joint_modeling_data.parquet'
OCC_MODEL_DIR = PROJ_ROOT / 'models' / 'occupancy'
ENERGY_MODEL_DIR = PROJ_ROOT / 'models' / 'energy'


@pytest.fixture(scope="module")
def joint_data():
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


@pytest.fixture(scope="module")
def energy_artifacts():
    return load_energy_model_artifacts(ENERGY_MODEL_DIR)


@pytest.fixture(scope="module")
def occ_artifacts():
    return load_occupancy_model_artifacts(OCC_MODEL_DIR)


def test_naive_baseline_calculation(joint_data):
    """Test 1: Verifies persistence and seasonal naive calculation definitions."""
    y = joint_data['south_wing_total_kwh_next_hour']
    
    # 1-step persistence: y_hat(t+1) = y(t)
    pers = compute_persistence_baseline(y)
    # Check that pers at index i equals y at index i-1
    assert pers.iloc[10] == y.iloc[9]
    assert pers.iloc[100] == y.iloc[99]
    
    # 24-step seasonal naive: y_hat(t+1) = y(t-24)
    snaive = compute_seasonal_naive_baseline(y, season_lag=24)
    assert snaive.iloc[25] == y.iloc[1]
    assert snaive.iloc[100] == y.iloc[76]
    
    # Evaluate on validation split
    val_mask = (joint_data.index >= '2018-12-01 00:00:00') & (joint_data.index <= '2019-01-10 23:00:00')
    y_val = y.loc[val_mask]
    pers_val = pers.loc[val_mask]
    
    metrics = evaluate_naive_baseline(y_val.values, pers_val.values)
    assert metrics['r2'] > 0.60
    assert metrics['mae'] < 4.0
    assert metrics['rmse'] < 5.0


def test_no_train_test_contamination(joint_data, occ_artifacts):
    """Test 2: Confirms zero train/test contamination in baseline and threshold derivations."""
    train_end = pd.Timestamp('2018-11-30 23:00:00')
    val_start = pd.Timestamp('2018-12-01 00:00:00')
    val_end = pd.Timestamp('2019-01-10 23:00:00')
    test_start = pd.Timestamp('2019-01-11 00:00:00')
    
    df_train = joint_data.loc[joint_data.index <= train_end]
    df_val = joint_data.loc[(joint_data.index >= val_start) & (joint_data.index <= val_end)]
    df_test = joint_data.loc[joint_data.index >= test_start]
    
    assert len(df_train.index.intersection(df_val.index)) == 0
    assert len(df_val.index.intersection(df_test.index)) == 0
    assert len(df_train.index.intersection(df_test.index)) == 0
    
    # Safety threshold must be derived exclusively from validation probabilities
    occ_model = occ_artifacts['model']
    occ_scaler = occ_artifacts['scaler']
    feat_names = occ_artifacts['feature_names']
    
    X_val = occ_scaler.transform(df_val[feat_names]) if occ_scaler else df_val[feat_names].values
    y_val_prob = occ_model.predict_proba(X_val)[:, 1]
    y_val_true = df_val['is_occupied_next_hour'].values
    
    # Compute threshold on validation ONLY
    thresh_val = select_control_safety_threshold(y_val_true, y_val_prob, max_acceptable_fnr=0.02)
    assert 0.20 <= thresh_val <= 0.35


def test_control_range_enforcement():
    """Test 3: Verifies boundary checking and clamping for candidate controls."""
    valid_controls = {
        'rtu_south_fan_spd_mean_next_hour': 65.0,
        'rtu_south_damper_pct_mean_next_hour': 45.0
    }
    is_valid, violations = validate_control_bounds(valid_controls)
    assert is_valid is True
    assert len(violations) == 0
    
    # Out of bounds cases
    invalid_controls = {
        'rtu_south_fan_spd_mean_next_hour': 25.0,  # Below min 40.0
        'rtu_south_damper_pct_mean_next_hour': 95.0 # Above max 90.0
    }
    is_valid, violations = validate_control_bounds(invalid_controls)
    assert is_valid is False
    assert len(violations) == 2
    
    # Test clamping
    clamped = clamp_controls(invalid_controls)
    assert clamped['rtu_south_fan_spd_mean_next_hour'] == 40.0
    assert clamped['rtu_south_damper_pct_mean_next_hour'] == 90.0


def test_threshold_safety_calculation(joint_data, occ_artifacts):
    """Test 4: Verifies safety threshold metrics and monotonic recall curve."""
    df_val = joint_data.loc[(joint_data.index >= '2018-12-01 00:00:00') & (joint_data.index <= '2019-01-10 23:00:00')]
    
    occ_model = occ_artifacts['model']
    occ_scaler = occ_artifacts['scaler']
    feat_names = occ_artifacts['feature_names']
    
    X_val = occ_scaler.transform(df_val[feat_names]) if occ_scaler else df_val[feat_names].values
    y_val_prob = occ_model.predict_proba(X_val)[:, 1]
    y_val_true = df_val['is_occupied_next_hour'].values
    
    curve = compute_occupancy_safety_curve(y_val_true, y_val_prob)
    
    # Monotonicity: recall should decrease or stay equal as threshold increases
    recalls = [pt['recall'] for pt in curve]
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1]
        
    # At recommended safety threshold (0.30), FNR must be <= 1.5% (recall >= 98.5%)
    point_30 = next(pt for pt in curve if abs(pt['threshold'] - RECOMMENDED_SAFETY_THRESHOLD) < 1e-3)
    assert point_30['false_negative_rate_unsafe'] <= 0.015
    assert point_30['recall'] >= 0.985
    
    # Conservative relationship: safety threshold must be strictly lower than F1-optimal threshold
    assert RECOMMENDED_SAFETY_THRESHOLD < RECOMMENDED_CLASS_THRESHOLD


def test_counterfactual_input_validation(joint_data, energy_artifacts):
    """Test 5: Verifies surrogate monotonicity and numerical stability under candidate controls."""
    model = energy_artifacts['model']
    scaler = energy_artifacts['scaler']
    features = energy_artifacts['feature_names']
    
    # 1. Inspect regression coefficients: fan speed must be positive, damper negative
    coef_dict = dict(zip(features, model.coef_))
    assert coef_dict['rtu_south_fan_spd_mean_next_hour'] > 2.0  # +2.6327
    assert coef_dict['rtu_south_damper_pct_mean_next_hour'] < -0.5 # -0.8826
    
    # 2. Select a representative occupied afternoon baseline from the test set
    test_mask = (joint_data.index >= '2019-01-11 00:00:00') & (joint_data.index <= '2019-02-21 09:00:00')
    df_test = joint_data.loc[test_mask]
    warm_occ_mask = (df_test['outdoor_temp_c'] >= 13.0) & (df_test['is_occupied_next_hour'] == 1)
    baseline_row = df_test.loc[warm_occ_mask].iloc[0][features]
    
    # Vary fan speed across candidate range [40, 50, 60, 70, 80, 90]
    fan_speeds = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    df_fan_res = evaluate_counterfactual_response(
        model, scaler, features, baseline_row, 'rtu_south_fan_spd_mean_next_hour', fan_speeds
    )
    
    # Physical monotonicity: power must increase strictly as fan speed increases
    preds = df_fan_res['predicted_kwh'].tolist()
    raw_preds = df_fan_res['raw_predicted_kwh'].tolist()
    for i in range(len(preds) - 1):
        assert preds[i + 1] > preds[i]
        assert raw_preds[i + 1] > raw_preds[i]
        
    # Non-negativity check
    assert all(p >= 0.0 for p in preds)
    
    # Economizer damper monotonicity: increasing outdoor air damper should reduce cooling power in warm weather
    damper_positions = [10.0, 20.0, 40.0, 60.0, 80.0]
    df_damper_res = evaluate_counterfactual_response(
        model, scaler, features, baseline_row, 'rtu_south_damper_pct_mean_next_hour', damper_positions
    )
    damper_raw = df_damper_res['raw_predicted_kwh'].tolist()
    for i in range(len(damper_raw) - 1):
        assert damper_raw[i + 1] < damper_raw[i]
