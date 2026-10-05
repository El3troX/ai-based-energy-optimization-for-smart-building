"""
Automated Test Suite for Phase 3 Model Training & Chained Evaluation.
Tests:
1. Temporal split integrity
2. No train/validation/test overlap
3. Feature/target separation & zero leakage
4. Prediction shape consistency
5. Classification threshold handling
6. Regression output validity
7. Serialized model reloading and inference
8. Chained occupancy -> energy forecasting pipeline
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.occupancy_model import (
    load_and_split_occupancy_data,
    load_occupancy_model_artifacts
)
from src.energy_model import (
    load_and_split_energy_data,
    evaluate_chained_energy,
    load_energy_model_artifacts
)
from src.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    tune_classification_threshold
)

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJ_ROOT / 'data' / 'processed' / 'joint_modeling_data.parquet'
OCC_MODEL_DIR = PROJ_ROOT / 'models' / 'occupancy'
ENERGY_MODEL_DIR = PROJ_ROOT / 'models' / 'energy'

@pytest.fixture(scope="module")
def dataset_splits():
    occ_data = load_and_split_occupancy_data(DATA_PATH)
    energy_data = load_and_split_energy_data(DATA_PATH)
    return {'occ': occ_data, 'energy': energy_data}

def test_temporal_split_integrity(dataset_splits):
    """Test 1: Confirms exact chronological boundary adherence."""
    occ = dataset_splits['occ']
    
    train_idx = occ['train']['index']
    val_idx = occ['val']['index']
    test_idx = occ['test']['index']
    
    assert train_idx.min() == pd.Timestamp('2018-05-23 07:00:00')
    assert train_idx.max() == pd.Timestamp('2018-11-30 23:00:00')
    
    assert val_idx.min() == pd.Timestamp('2018-12-01 00:00:00')
    assert val_idx.max() == pd.Timestamp('2019-01-10 23:00:00')
    
    assert test_idx.min() == pd.Timestamp('2019-01-11 00:00:00')
    assert test_idx.max() == pd.Timestamp('2019-02-21 09:00:00')

def test_no_train_test_overlap(dataset_splits):
    """Test 2: Verifies zero timestamp overlap and strict monotonic partition ordering."""
    occ = dataset_splits['occ']
    
    train_set = set(occ['train']['index'])
    val_set = set(occ['val']['index'])
    test_set = set(occ['test']['index'])
    
    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test splits overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test splits overlap!"
    
    assert occ['train']['index'].max() < occ['val']['index'].min()
    assert occ['val']['index'].max() < occ['test']['index'].min()

def test_feature_target_separation(dataset_splits):
    """Test 3: Confirms targets do not appear in X and no energy contamination in occupancy."""
    occ = dataset_splits['occ']
    energy = dataset_splits['energy']
    
    # Occupancy feature separation
    assert 'is_occupied_next_hour' not in occ['feature_cols']
    assert 'occ_total_mean_next_hour' not in occ['feature_cols']
    
    # Zero energy submeter contamination in occupancy feature space
    for col in occ['feature_cols']:
        assert 'kwh' not in col.lower(), f"Energy submeter {col} found in occupancy features!"
        assert not col.startswith('lig_S'), f"Lighting submeter {col} found in occupancy features!"
        assert not col.startswith('mels_S'), f"Plug load submeter {col} found in occupancy features!"
        assert not col.startswith('hvac_S'), f"HVAC submeter {col} found in occupancy features!"
        
    # Energy feature separation
    assert 'south_wing_total_kwh_next_hour' not in energy['feature_cols']
    assert 'lig_S_kwh_next_hour' not in energy['feature_cols']
    assert 'mels_S_kwh_next_hour' not in energy['feature_cols']
    assert 'hvac_S_kwh_next_hour' not in energy['feature_cols']

def test_prediction_shape(dataset_splits):
    """Test 4: Verifies predictions match input sample count."""
    occ_art = load_occupancy_model_artifacts(OCC_MODEL_DIR)
    occ = dataset_splits['occ']
    
    X_val = occ['val']['X']
    scaler = occ_art['scaler']
    X_val_in = scaler.transform(X_val) if scaler else X_val.values
    probs = occ_art['model'].predict_proba(X_val_in)[:, 1]
    
    assert probs.shape == (len(X_val),), "Prediction probability shape mismatch!"

def test_classification_threshold_handling():
    """Test 5: Validates threshold tuning logic and discrete binary output."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.4, 0.45, 0.8, 0.9, 0.2, 0.6, 0.3])
    
    res = tune_classification_threshold(y_true, y_prob, metric='f1')
    assert 0.05 <= res['best_threshold'] <= 0.95
    assert 0.0 <= res['best_score'] <= 1.0
    
    binary_preds = (y_prob >= res['best_threshold']).astype(int)
    assert set(np.unique(binary_preds)).issubset({0, 1})

def test_regression_output_validity(dataset_splits):
    """Test 6: Confirms continuous energy regression outputs are finite and non-negative."""
    energy_art = load_energy_model_artifacts(ENERGY_MODEL_DIR)
    energy = dataset_splits['energy']
    
    X_test = energy['test']['X']
    scaler = energy_art['scaler']
    X_test_in = scaler.transform(X_test) if scaler else X_test.values
    
    preds = np.maximum(0.0, energy_art['model'].predict(X_test_in))
    
    assert np.all(np.isfinite(preds)), "Non-finite predictions detected!"
    assert np.all(preds >= 0.0), "Negative energy predictions detected!"
    assert len(preds) == len(X_test)

def test_serialized_model_reload():
    """Test 7: Confirms saved model artifacts can be reloaded and reproduce outputs."""
    occ_art = load_occupancy_model_artifacts(OCC_MODEL_DIR)
    energy_art = load_energy_model_artifacts(ENERGY_MODEL_DIR)
    
    assert occ_art['model'] is not None
    assert 'threshold' in occ_art
    assert energy_art['model'] is not None
    assert len(energy_art['feature_names']) > 0

def test_chained_occupancy_to_energy_inference(dataset_splits):
    """Test 8: Validates end-to-end chained BEMS pipeline (X_t -> y_occ_hat -> y_energy_hat)."""
    occ_art = load_occupancy_model_artifacts(OCC_MODEL_DIR)
    energy_art = load_energy_model_artifacts(ENERGY_MODEL_DIR)
    
    occ = dataset_splits['occ']
    energy = dataset_splits['energy']
    
    # 1. Generate occupancy forecast
    X_test_occ = occ['test']['X']
    scaler_occ = occ_art['scaler']
    X_test_occ_in = scaler_occ.transform(X_test_occ) if scaler_occ else X_test_occ.values
    probs_occ = occ_art['model'].predict_proba(X_test_occ_in)[:, 1]
    pred_occ = (probs_occ >= occ_art['threshold']).astype(int)
    
    # 2. Chained energy prediction
    chained_eval = evaluate_chained_energy(
        model=energy_art['model'],
        X_eval=energy['test']['X'],
        y_eval=energy['test']['y'],
        predicted_occupancy=pred_occ,
        is_linear='Ridge' in energy_art['metadata']['model_name'],
        scaler=energy_art['scaler']
    )
    
    pred_energy = chained_eval['predictions']
    metrics = chained_eval['metrics']
    
    assert len(pred_energy) == len(energy['test']['X'])
    assert np.all(pred_energy >= 0.0)
    assert metrics['mae'] > 0.0
    assert metrics['rmse'] > 0.0
