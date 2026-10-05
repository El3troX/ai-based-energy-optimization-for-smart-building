"""
Robustness and Safety Evaluation Suite for BEMS Control Optimization.
Provides utilities for:
- Naive baseline calculations (Persistence and Seasonal Naive)
- Occupancy control-safety threshold calibration (FNR minimization)
- Control support validation and bound enforcement
- Counterfactual surrogate sensitivity & monotonicity checks
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

from src.evaluation import compute_regression_metrics

# Standard empirical support bounds derived from Phase 3.5 audit
CONTROL_BOUNDS = {
    'rtu_south_fan_spd_mean_next_hour': {'min': 40.0, 'max': 90.0},
    'rtu_south_damper_pct_mean_next_hour': {'min': 10.0, 'max': 90.0}
}

RECOMMENDED_SAFETY_THRESHOLD = 0.30
RECOMMENDED_CLASS_THRESHOLD = 0.51
RECOMMENDED_BASELOAD_FLOOR = 6.50


def compute_persistence_baseline(series: pd.Series) -> pd.Series:
    """
    Computes 1-step persistence baseline: y_hat(t+1) = y(t).
    """
    return series.shift(1)


def compute_seasonal_naive_baseline(series: pd.Series, season_lag: int = 24) -> pd.Series:
    """
    Computes seasonal naive baseline: y_hat(t+1) = y(t+1 - season_lag).
    """
    return series.shift(season_lag)


def evaluate_naive_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates baseline predictions, dropping NaNs if any.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        raise ValueError("No valid overlapping observations for baseline evaluation.")
    return compute_regression_metrics(y_true[mask], y_pred[mask])


def compute_occupancy_safety_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Computes threshold, recall, specificity, FNR, FPR, precision, and F1 across thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
        
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    curve = []
    for t in thresholds:
        t_val = round(float(t), 4)
        y_pred = (y_prob >= t_val).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0
        
        curve.append({
            'threshold': t_val,
            'recall': round(float(recall), 4),
            'specificity': round(float(specificity), 4),
            'false_negative_rate_unsafe': round(float(fnr), 4),
            'false_positive_rate_over_conservative': round(float(fpr), 4),
            'precision': round(float(prec), 4),
            'f1': round(float(f1), 4)
        })
    return curve


def select_control_safety_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_acceptable_fnr: float = 0.015,
    thresholds: Optional[np.ndarray] = None
) -> float:
    """
    Selects the maximum operating threshold that satisfies FNR <= max_acceptable_fnr.
    Ensures safe operation by preventing false-unoccupied events.
    """
    curve = compute_occupancy_safety_curve(y_true, y_prob, thresholds=thresholds)
    valid_candidates = [
        item for item in curve 
        if item['false_negative_rate_unsafe'] <= max_acceptable_fnr
    ]
    if not valid_candidates:
        # If no threshold strictly satisfies, choose the lowest available threshold
        return float(curve[0]['threshold'])
    
    # Pick the highest threshold satisfying safety constraint (maximizes specificity/energy saving while safe)
    best = max(valid_candidates, key=lambda x: x['threshold'])
    return float(best['threshold'])


def validate_control_bounds(
    controls: Dict[str, float],
    bounds: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[bool, List[str]]:
    """
    Checks if proposed candidate controls are within empirical support bounds.
    Returns (is_valid, list_of_violations).
    """
    if bounds is None:
        bounds = CONTROL_BOUNDS
        
    violations = []
    for var_name, val in controls.items():
        if var_name in bounds:
            b_min = bounds[var_name]['min']
            b_max = bounds[var_name]['max']
            if val < b_min or val > b_max:
                violations.append(
                    f"Control '{var_name}' value {val} out of bounds [{b_min}, {b_max}]"
                )
    return len(violations) == 0, violations


def clamp_controls(
    controls: Dict[str, float],
    bounds: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """
    Clamps candidate controls to the allowed support bounds.
    """
    if bounds is None:
        bounds = CONTROL_BOUNDS
        
    clamped = {}
    for var_name, val in controls.items():
        if var_name in bounds:
            b_min = bounds[var_name]['min']
            b_max = bounds[var_name]['max']
            clamped[var_name] = float(np.clip(val, b_min, b_max))
        else:
            clamped[var_name] = val
    return clamped


def evaluate_counterfactual_response(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    baseline_features: pd.Series,
    control_var: str,
    test_values: List[float],
    clamp_zero: bool = True
) -> pd.DataFrame:
    """
    Computes surrogate-predicted energy response across a range of values for a single control variable.
    """
    records = []
    base_dict = baseline_features.to_dict()
    
    for val in test_values:
        row = base_dict.copy()
        row[control_var] = val
        X_df = pd.DataFrame([row])[feature_names]
        X_scaled = scaler.transform(X_df) if scaler is not None else X_df.values
        raw_pred = float(model.predict(X_scaled)[0])
        pred_kwh = float(np.maximum(0.0, raw_pred)) if clamp_zero else raw_pred
        records.append({
            'control_var': control_var,
            'control_value': val,
            'predicted_kwh': pred_kwh,
            'raw_predicted_kwh': raw_pred
        })
    return pd.DataFrame(records)
