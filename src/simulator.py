"""
BEMS Building Operational Simulator (Phase 4).
Simulates baseline vs counterfactual optimized operations across chronological partitions.

Enforces:
- Separation of actual historical energy, baseline model predictions, and counterfactual predictions
- Stratified scenario evaluation (Baseline, HVAC-only, HVAC + Lighting, Full Safeguarded)
- Safety & ramp constraint ablation studies
- Multi-regime error and confidence reporting
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON
)
from src.optimizer import (
    load_optimization_models,
    optimize_hour
)


def simulate_period(
    df: pd.DataFrame,
    occ_model_artifacts: Dict[str, Any],
    energy_model_artifacts: Dict[str, Any],
    safety_threshold: float = T_SAFETY,
    classification_threshold: float = T_CLASS,
    enforce_ramps: bool = True,
    enforce_comfort: bool = True,
    min_improvement_epsilon: float = MIN_IMPROVEMENT_EPSILON,
    include_lighting: bool = False
) -> pd.DataFrame:
    """
    Executes hour-by-hour counterfactual simulation across a given DataFrame partition.
    Returns simulation results as a DataFrame indexed by timestamp.
    """
    results = []
    
    for ts, row in df.iterrows():
        row_named = row.copy()
        row_named.name = ts
        
        hour_res = optimize_hour(
            row=row_named,
            occ_model_artifacts=occ_model_artifacts,
            energy_model_artifacts=energy_model_artifacts,
            safety_threshold=safety_threshold,
            classification_threshold=classification_threshold,
            enforce_ramps=enforce_ramps,
            enforce_comfort=enforce_comfort,
            min_improvement_epsilon=min_improvement_epsilon,
            include_lighting=include_lighting
        )
        
        # Merge actual historical values if available for reference
        hour_res['actual_total_energy'] = round(float(row.get('south_wing_total_kwh_next_hour', np.nan)), 4)
        hour_res['outdoor_temp_c'] = round(float(row.get('outdoor_temp_c', np.nan)), 2)
        hour_res['actual_occupancy_state'] = int(row.get('is_occupied_next_hour', 0))
        results.append(hour_res)
        
    res_df = pd.DataFrame(results)
    if 'timestamp' in res_df.columns:
        res_df['timestamp'] = pd.to_datetime(res_df['timestamp'])
        res_df = res_df.set_index('timestamp').sort_index()
        
    return res_df


def summarize_simulation(sim_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for a simulation experiment.
    """
    n_hours = len(sim_df)
    opt_mask = sim_df['action_reason'].str.startswith('Identified')
    n_opt = int(opt_mask.sum())
    n_no_change = n_hours - n_opt
    
    total_savings_kwh = float(sim_df['estimated_energy_saving'].sum())
    mean_delta_opt = float(sim_df.loc[opt_mask, 'raw_delta_energy'].mean()) if n_opt > 0 else 0.0
    median_delta_opt = float(sim_df.loc[opt_mask, 'raw_delta_energy'].median()) if n_opt > 0 else 0.0
    pct_improved = (n_opt / n_hours * 100.0) if n_hours > 0 else 0.0
    
    total_cost_saving = float(sim_df['estimated_cost_saving'].sum())
    total_co2_saving = float(sim_df['estimated_co2_reduction'].sum())
    
    baseline_pred_sum = float(sim_df['baseline_predicted_energy'].sum())
    overall_pct_saving = (total_savings_kwh / baseline_pred_sum * 100.0) if baseline_pred_sum > 0 else 0.0
    
    return {
        'hours_evaluated': n_hours,
        'hours_optimized': n_opt,
        'hours_no_change': n_no_change,
        'pct_hours_improved': round(pct_improved, 2),
        'total_estimated_energy_savings_kwh': round(total_savings_kwh, 2),
        'mean_delta_per_optimized_hour_kwh': round(mean_delta_opt, 4),
        'median_delta_per_optimized_hour_kwh': round(median_delta_opt, 4),
        'overall_estimated_energy_savings_pct': round(overall_pct_saving, 2),
        'total_estimated_cost_savings_usd': round(total_cost_saving, 2),
        'total_estimated_co2_reduction_kg': round(total_co2_saving, 2)
    }


def compute_regime_breakdown(sim_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Decomposes simulation results across occupancy states and outdoor temperature regimes.
    """
    breakdown = {}
    
    # 1. Occupancy breakdown
    for occ_state, label in [(1, 'Occupied'), (0, 'Unoccupied')]:
        sub = sim_df[sim_df['actual_occupancy_state'] == occ_state]
        breakdown[f'occupancy_{label.lower()}'] = {
            'regime': label,
            'hours': len(sub),
            'hours_optimized': int((sub['estimated_energy_saving'] > 0).sum()),
            'savings_kwh': round(float(sub['estimated_energy_saving'].sum()), 2),
            'cost_savings_usd': round(float(sub['estimated_cost_saving'].sum()), 2),
            'co2_reduction_kg': round(float(sub['estimated_co2_reduction'].sum()), 2)
        }
        
    # 2. Outdoor temperature regimes
    temp_regimes = [
        ('Cold (<10C)', sim_df['outdoor_temp_c'] < 10.0),
        ('Mild (10-16C)', (sim_df['outdoor_temp_c'] >= 10.0) & (sim_df['outdoor_temp_c'] <= 16.0)),
        ('Warm (>16C)', sim_df['outdoor_temp_c'] > 16.0)
    ]
    for label, mask in temp_regimes:
        sub = sim_df[mask]
        breakdown[f'temp_{label[:4].lower().strip()}'] = {
            'regime': label,
            'hours': len(sub),
            'hours_optimized': int((sub['estimated_energy_saving'] > 0).sum()),
            'savings_kwh': round(float(sub['estimated_energy_saving'].sum()), 2),
            'cost_savings_usd': round(float(sub['estimated_cost_saving'].sum()), 2),
            'co2_reduction_kg': round(float(sub['estimated_co2_reduction'].sum()), 2)
        }
        
    return breakdown


def run_experiment_scenarios(
    df_test: pd.DataFrame,
    occ_artifacts: Dict[str, Any],
    energy_artifacts: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the four required experimental scenarios:
    A. Baseline (No optimization)
    B. HVAC-only optimization
    C. HVAC + Lighting rule
    D. Full Safeguarded Optimizer
    """
    print("Executing Scenario A: Baseline (No Optimization)...")
    res_a = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        min_improvement_epsilon=1e6 # Disables optimization by setting impossible improvement threshold
    )
    
    print("Executing Scenario B: HVAC-Only Optimization...")
    res_b = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        include_lighting=False, enforce_ramps=True, enforce_comfort=True
    )
    
    print("Executing Scenario C: HVAC + Lighting Rule...")
    res_c = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        include_lighting=True, enforce_ramps=True, enforce_comfort=True
    )
    
    print("Executing Scenario D: Full Safeguarded Optimizer...")
    res_d = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        safety_threshold=T_SAFETY, enforce_ramps=True, enforce_comfort=True,
        include_lighting=True
    )
    
    return {
        'Scenario A (Baseline)': {'summary': summarize_simulation(res_a), 'df': res_a},
        'Scenario B (HVAC-Only)': {'summary': summarize_simulation(res_b), 'df': res_b},
        'Scenario C (HVAC + Lighting)': {'summary': summarize_simulation(res_c), 'df': res_c},
        'Scenario D (Full Safeguarded)': {'summary': summarize_simulation(res_d), 'df': res_d}
    }


def run_ablation_study(
    df_test: pd.DataFrame,
    occ_artifacts: Dict[str, Any],
    energy_artifacts: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the four required ablation configurations:
    A. Without occupancy safety threshold (T = 0.51)
    B. With occupancy safety threshold (T = 0.30)
    C. Without ramp-rate protection (enforce_ramps = False)
    D. With ramp-rate protection (enforce_ramps = True)
    """
    print("Executing Ablation A: Without Safety Gate (T=0.51)...")
    res_a = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        safety_threshold=0.51, enforce_ramps=True
    )
    
    print("Executing Ablation B: With Safety Gate (T=0.30)...")
    res_b = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        safety_threshold=0.30, enforce_ramps=True
    )
    
    print("Executing Ablation C: Without Ramp-Rate Protection...")
    res_c = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        safety_threshold=0.30, enforce_ramps=False
    )
    
    print("Executing Ablation D: With Ramp-Rate Protection...")
    res_d = simulate_period(
        df_test, occ_artifacts, energy_artifacts,
        safety_threshold=0.30, enforce_ramps=True
    )
    
    # Calculate safety metrics: potentially unsafe actions during actual occupied hours
    for name, res_df in [('A_no_safety_gate', res_a), ('B_with_safety_gate', res_b)]:
        unsafe_mask = (res_df['actual_occupancy_state'] == 1) & (res_df['control_safety_gate'] == 'SETBACK_ELIGIBLE')
        n_unsafe = int(unsafe_mask.sum())
        print(f"Safety Analysis - {name}: {n_unsafe} hours where setback was eligible during actual occupancy")
        
    return {
        'Ablation A (T=0.51, No Safety Gate)': {'summary': summarize_simulation(res_a), 'df': res_a},
        'Ablation B (T=0.30, With Safety Gate)': {'summary': summarize_simulation(res_b), 'df': res_b},
        'Ablation C (No Ramp Protection)': {'summary': summarize_simulation(res_c), 'df': res_c},
        'Ablation D (With Ramp Protection)': {'summary': summarize_simulation(res_d), 'df': res_d}
    }
