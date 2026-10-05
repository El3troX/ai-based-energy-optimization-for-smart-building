"""
BEMS Counterfactual Optimization Engine (Phase 4).
Provides model-based scenario search over HVAC and lighting setpoints to identify
admissible operating configurations that minimize model-predicted energy under
occupancy safety, equipment ramp, and comfort constraints.

IMPORTANT MODELING NOTE:
This is a research simulation using the Ridge regression model as an empirical surrogate.
Optimization operates strictly on the LOCAL DIFFERENTIAL objective:
    Delta E(u) = E_hat(u_candidate) - E_hat(u_baseline)
under identical exogenous context.
"""
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON,
    BASELOAD_ENERGY_FLOOR,
    LIGHTING_STANDBY_KWH,
    SEARCH_GRIDS,
    COMFORT_RULES
)
from src.occupancy_model import load_occupancy_model_artifacts
from src.energy_model import load_energy_model_artifacts


def load_optimization_models(
    occupancy_dir: str = 'models/occupancy',
    energy_dir: str = 'models/energy'
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads and returns frozen serialized occupancy and energy model artifacts.
    """
    occ_art = load_occupancy_model_artifacts(occupancy_dir)
    energy_art = load_energy_model_artifacts(energy_dir)
    return occ_art, energy_art


def apply_occupancy_safety_gate(prob_occ: float, safety_threshold: float = T_SAFETY) -> bool:
    """
    Applies the control-safety threshold to determine if upcoming period is control-protected occupied.
    Returns True if occupied (safety protected), False if setback allowed.
    """
    return bool(prob_occ >= safety_threshold)


def validate_control_candidate(
    candidate_fan: float,
    candidate_damper: float,
    baseline_fan: float,
    baseline_damper: float,
    is_control_occupied: bool,
    enforce_ramps: bool = True,
    enforce_comfort: bool = True
) -> Tuple[bool, str]:
    """
    Validates candidate controls against:
    1. Historical support boundaries [40, 90] fan, [10, 90] damper.
    2. Inter-hour ramp rate limits (max 15% fan, 30% damper).
    3. Comfort and operating mode ventilation constraints.
    
    Returns (is_valid, reason_str).
    """
    fan_bounds = CONTROL_BOUNDS['rtu_south_fan_spd_mean_next_hour']
    damper_bounds = CONTROL_BOUNDS['rtu_south_damper_pct_mean_next_hour']
    
    # 1. Historical support bounds
    if candidate_fan < fan_bounds['min'] or candidate_fan > fan_bounds['max']:
        return False, f"Fan speed {candidate_fan}% outside supported range [{fan_bounds['min']}, {fan_bounds['max']}]"
    if candidate_damper < damper_bounds['min'] or candidate_damper > damper_bounds['max']:
        return False, f"Damper {candidate_damper}% outside supported range [{damper_bounds['min']}, {damper_bounds['max']}]"
        
    # 2. Ramp rate limits
    if enforce_ramps:
        # Effective max delta allows transition into support if baseline was out-of-support
        allowed_fan_delta = max(
            RAMP_LIMITS['fan_speed_max_delta'],
            max(0.0, fan_bounds['min'] - baseline_fan),
            max(0.0, baseline_fan - fan_bounds['max'])
        )
        if abs(candidate_fan - baseline_fan) > allowed_fan_delta + 1e-4:
            return False, f"Fan ramp |{candidate_fan} - {baseline_fan}| > {allowed_fan_delta}%"
            
        allowed_damper_delta = max(
            RAMP_LIMITS['damper_pct_max_delta'],
            max(0.0, damper_bounds['min'] - baseline_damper),
            max(0.0, baseline_damper - damper_bounds['max'])
        )
        if abs(candidate_damper - baseline_damper) > allowed_damper_delta + 1e-4:
            return False, f"Damper ramp |{candidate_damper} - {baseline_damper}| > {allowed_damper_delta}%"

    # 3. Comfort and operational mode constraints
    if enforce_comfort:
        if is_control_occupied:
            # When occupied, preserve minimum ventilation airflow and outdoor air
            if candidate_fan < COMFORT_RULES['occupied']['min_fan_speed']:
                return False, f"Occupied fan speed {candidate_fan}% below comfort min {COMFORT_RULES['occupied']['min_fan_speed']}%"
            if candidate_damper < COMFORT_RULES['occupied']['min_damper_pct']:
                return False, f"Occupied damper {candidate_damper}% below comfort min {COMFORT_RULES['occupied']['min_damper_pct']}%"
        else:
            # When unoccupied, prevent unnecessary high ventilation
            if candidate_fan > COMFORT_RULES['unoccupied']['max_fan_speed']:
                return False, f"Unoccupied fan speed {candidate_fan}% exceeds setback max {COMFORT_RULES['unoccupied']['max_fan_speed']}%"
            if candidate_damper < COMFORT_RULES['unoccupied']['min_damper_pct']:
                return False, f"Unoccupied damper {candidate_damper}% below minimum {COMFORT_RULES['unoccupied']['min_damper_pct']}%"

    return True, "VALID"


def generate_candidate_grid(
    baseline_fan: float,
    baseline_damper: float,
    is_control_occupied: bool,
    enforce_ramps: bool = True,
    enforce_comfort: bool = True
) -> List[Tuple[float, float]]:
    """
    Generates all valid (fan, damper) candidate control pairs from the search grid.
    """
    fan_grid = SEARCH_GRIDS['fan_speed']
    damper_grid = SEARCH_GRIDS['damper_pct']
    
    valid_candidates = []
    for fan in fan_grid:
        for damper in damper_grid:
            is_valid, _ = validate_control_candidate(
                fan, damper, baseline_fan, baseline_damper,
                is_control_occupied, enforce_ramps, enforce_comfort
            )
            if is_valid:
                valid_candidates.append((fan, damper))
                
    return valid_candidates


def compute_surrogate_confidence(
    outdoor_temp_c: float,
    fan_speed: float,
    damper_pct: float
) -> str:
    """
    Assigns an explicit validity and confidence flag to the surrogate prediction.
    """
    if outdoor_temp_c < 10.0:
        return "LOW_SURROGATE_CONFIDENCE (Cold Regime <10C)"
    if fan_speed <= 40.0 or fan_speed >= 90.0 or damper_pct <= 10.0 or damper_pct >= 90.0:
        return "BOUNDARY_WARNING"
    return "NORMAL_CONFIDENCE"


def predict_candidate_energy(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    base_features: Dict[str, Any],
    candidate_fan: float,
    candidate_damper: float,
    pred_occ_next_hour: int
) -> float:
    """
    Evaluates the Ridge surrogate for candidate controls and predicted occupancy state.
    Returns raw model-predicted kWh.
    """
    x_row = np.array([[base_features[f] for f in feature_names]], dtype=float)
    fan_idx = feature_names.index('rtu_south_fan_spd_mean_next_hour')
    damper_idx = feature_names.index('rtu_south_damper_pct_mean_next_hour')
    occ_idx = feature_names.index('is_occupied_next_hour')
    x_row[0, fan_idx] = candidate_fan
    x_row[0, damper_idx] = candidate_damper
    x_row[0, occ_idx] = pred_occ_next_hour
    X_df = pd.DataFrame(x_row, columns=feature_names)
    X_scaled = scaler.transform(X_df) if scaler is not None else X_df.values
    return float(model.predict(X_scaled)[0])


def optimize_hour(
    row: pd.Series,
    occ_model_artifacts: Dict[str, Any],
    energy_model_artifacts: Dict[str, Any],
    safety_threshold: float = T_SAFETY,
    classification_threshold: float = T_CLASS,
    enforce_ramps: bool = True,
    enforce_comfort: bool = True,
    min_improvement_epsilon: float = MIN_IMPROVEMENT_EPSILON,
    include_lighting: bool = False
) -> Dict[str, Any]:
    """
    Performs counterfactual scenario evaluation for a single hour.
    
    Returns structured optimization record with baseline, recommended controls,
    differential predicted savings, cost, emissions, and validity metadata.
    """
    occ_model = occ_model_artifacts['model']
    occ_scaler = occ_model_artifacts['scaler']
    occ_features = occ_model_artifacts['feature_names']
    
    energy_model = energy_model_artifacts['model']
    energy_scaler = energy_model_artifacts['scaler']
    energy_features = energy_model_artifacts['feature_names']
    
    # 1. Occupancy Prediction & Safety Gating
    X_occ = np.array([[row[f] for f in occ_features]], dtype=float)
    X_occ_df = pd.DataFrame(X_occ, columns=occ_features)
    X_occ_scaled = occ_scaler.transform(X_occ_df) if occ_scaler else X_occ_df.values
    prob_occ = float(occ_model.predict_proba(X_occ_scaled)[0, 1])
    
    is_occupied_class = int(prob_occ >= classification_threshold)
    is_control_occupied = apply_occupancy_safety_gate(prob_occ, safety_threshold)
    
    # 2. Baseline Controls & Exogenous Feature Preparation
    baseline_fan = float(row['rtu_south_fan_spd_mean_next_hour'])
    baseline_damper = float(row['rtu_south_damper_pct_mean_next_hour'])
    outdoor_temp = float(row['outdoor_temp_c'])
    
    base_energy_features = {f: row[f] for f in energy_features if f in row}
    
    # 3. Evaluate Baseline Through Energy Surrogate
    baseline_raw_kwh = predict_candidate_energy(
        energy_model, energy_scaler, energy_features,
        base_energy_features, baseline_fan, baseline_damper,
        pred_occ_next_hour=is_occupied_class
    )
    baseline_display_kwh = max(BASELOAD_ENERGY_FLOOR, baseline_raw_kwh)
    
    # 4. Generate & Search Admissible Candidate Controls (Vectorized)
    candidates = generate_candidate_grid(
        baseline_fan, baseline_damper, is_control_occupied,
        enforce_ramps=enforce_ramps, enforce_comfort=enforce_comfort
    )
    
    best_cand = None
    best_delta = 0.0
    best_raw_kwh = baseline_raw_kwh
    
    if len(candidates) > 0:
        x_base = np.array([base_energy_features[f] for f in energy_features], dtype=float)
        fan_idx = energy_features.index('rtu_south_fan_spd_mean_next_hour')
        damper_idx = energy_features.index('rtu_south_damper_pct_mean_next_hour')
        occ_idx = energy_features.index('is_occupied_next_hour')
        
        X_mat = np.tile(x_base, (len(candidates), 1))
        X_mat[:, occ_idx] = is_occupied_class
        X_mat[:, fan_idx] = [c[0] for c in candidates]
        X_mat[:, damper_idx] = [c[1] for c in candidates]
        
        X_mat_df = pd.DataFrame(X_mat, columns=energy_features)
        X_scaled = energy_scaler.transform(X_mat_df) if energy_scaler is not None else X_mat_df.values
        cand_raw_kwhs = energy_model.predict(X_scaled)
        
        # Optional Lighting Rule integration
        lighting_delta = 0.0
        if include_lighting and (not is_control_occupied):
            baseline_lig = float(row.get('lig_S_kwh', LIGHTING_STANDBY_KWH))
            if baseline_lig > LIGHTING_STANDBY_KWH:
                lighting_delta = LIGHTING_STANDBY_KWH - baseline_lig
                
        hvac_deltas = cand_raw_kwhs - baseline_raw_kwh
        total_deltas = hvac_deltas + lighting_delta
        
        min_idx = int(np.argmin(total_deltas))
        if total_deltas[min_idx] < best_delta:
            best_delta = float(total_deltas[min_idx])
            best_cand = candidates[min_idx]
            best_raw_kwh = float(cand_raw_kwhs[min_idx])
            
    # 5. Baseline Preservation Logic (NO CHANGE Gate)
    # Require estimated saving >= min_improvement_epsilon to avoid trivial micro-adjustments
    if best_cand is not None and best_delta <= -min_improvement_epsilon:
        action = "OPTIMIZE"
        rec_fan = best_cand[0]
        rec_damper = best_cand[1]
        raw_delta = best_delta
        opt_raw_kwh = best_raw_kwh
        saving_kwh = -best_delta
        reason = f"Identified safe candidate reducing energy by {saving_kwh:.2f} kWh (fan={rec_fan}%, damper={rec_damper}%)"
    else:
        action = "NO_CHANGE"
        rec_fan = baseline_fan
        rec_damper = baseline_damper
        raw_delta = 0.0
        opt_raw_kwh = baseline_raw_kwh
        saving_kwh = 0.0
        reason = f"No candidate exceeded improvement threshold epsilon={min_improvement_epsilon:.2f} kWh"
        
    opt_display_kwh = max(BASELOAD_ENERGY_FLOOR, opt_raw_kwh)
    saving_pct = (saving_kwh / baseline_display_kwh * 100.0) if baseline_display_kwh > 0 else 0.0
    cost_saving = saving_kwh * ELECTRICITY_TARIFF
    co2_saving = saving_kwh * EMISSION_FACTOR
    
    confidence_flag = compute_surrogate_confidence(outdoor_temp, rec_fan, rec_damper)
    
    return {
        'timestamp': row.name if hasattr(row, 'name') else None,
        'occupancy_probability': round(prob_occ, 4),
        'occupancy_state': is_occupied_class,
        'control_safety_gate': 'PROTECTED_OCCUPIED' if is_control_occupied else 'SETBACK_ELIGIBLE',
        'baseline_fan': round(baseline_fan, 2),
        'baseline_damper': round(baseline_damper, 2),
        'recommended_fan': round(rec_fan, 2),
        'recommended_damper': round(rec_damper, 2),
        'baseline_predicted_energy': round(baseline_display_kwh, 4),
        'optimized_predicted_energy': round(opt_display_kwh, 4),
        'raw_baseline_energy': round(baseline_raw_kwh, 4),
        'raw_optimized_energy': round(opt_raw_kwh, 4),
        'raw_delta_energy': round(raw_delta, 4),
        'optimized_delta_energy': round(raw_delta, 4),
        'estimated_energy_saving': round(saving_kwh, 4),
        'estimated_saving_percent': round(saving_pct, 2),
        'estimated_cost_saving': round(cost_saving, 4),
        'estimated_co2_reduction': round(co2_saving, 4),
        'action_reason': reason,
        'constraint_status': 'ENFORCED',
        'confidence_flag': confidence_flag
    }
