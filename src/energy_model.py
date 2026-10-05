"""
Stage 2: Energy Forecasting Models (t -> t+1).
Implements primary building energy regression (south_wing_total_kwh_next_hour)
and diagnostic submeter regression (hvac_S, lig_S, mels_S) supporting both
Oracle (true occupancy) and Chained (predicted occupancy) forecasting regimes.
"""
import os
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.evaluation import compute_regression_metrics

ENERGY_EXCLUDED_COLS = [
    'timestamp',
    'south_wing_total_kwh_next_hour',
    'lig_S_kwh_next_hour',
    'mels_S_kwh_next_hour',
    'hvac_S_kwh_next_hour',
    'occ_total_mean_next_hour'
    # Note: is_occupied_next_hour is KEPT as the occupancy input feature!
]

def load_and_split_energy_data(data_path):
    """
    Loads joint parquet dataset and constructs frozen Train, Validation, and Test splits.
    """
    df = pd.read_parquet(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
    train_mask = (df.index >= '2018-05-23 07:00:00') & (df.index <= '2018-11-30 23:00:00')
    val_mask = (df.index >= '2018-12-01 00:00:00') & (df.index <= '2019-01-10 23:00:00')
    test_mask = (df.index >= '2019-01-11 00:00:00') & (df.index <= '2019-02-21 09:00:00')
    
    feature_cols = [c for c in df.columns if c not in ENERGY_EXCLUDED_COLS]
    
    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, 'south_wing_total_kwh_next_hour'].values
    
    X_val = df.loc[val_mask, feature_cols].copy()
    y_val = df.loc[val_mask, 'south_wing_total_kwh_next_hour'].values
    
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, 'south_wing_total_kwh_next_hour'].values
    
    # Submeter targets for diagnostics
    submeters = {
        'hvac': {
            'train': df.loc[train_mask, 'hvac_S_kwh_next_hour'].values,
            'val': df.loc[val_mask, 'hvac_S_kwh_next_hour'].values,
            'test': df.loc[test_mask, 'hvac_S_kwh_next_hour'].values
        },
        'lig': {
            'train': df.loc[train_mask, 'lig_S_kwh_next_hour'].values,
            'val': df.loc[val_mask, 'lig_S_kwh_next_hour'].values,
            'test': df.loc[test_mask, 'lig_S_kwh_next_hour'].values
        },
        'mels': {
            'train': df.loc[train_mask, 'mels_S_kwh_next_hour'].values,
            'val': df.loc[val_mask, 'mels_S_kwh_next_hour'].values,
            'test': df.loc[test_mask, 'mels_S_kwh_next_hour'].values
        }
    }
    
    return {
        'feature_cols': feature_cols,
        'train': {'X': X_train, 'y': y_train, 'index': X_train.index},
        'val': {'X': X_val, 'y': y_val, 'index': X_val.index},
        'test': {'X': X_test, 'y': y_test, 'index': X_test.index},
        'submeters': submeters
    }

def get_candidate_energy_regressors(random_state=42):
    """
    Returns candidate regression models for total building energy forecasting.
    """
    return {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1)
    }

def train_and_eval_energy_regressors(X_train, y_train, X_val, y_val, feature_cols, random_state=42):
    """
    Fits all candidate regressors on Train and evaluates on Validation (Oracle regime).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = get_candidate_energy_regressors(random_state=random_state)
    results = {}
    
    for name, model in models.items():
        is_linear = 'Ridge' in name
        X_tr = X_train_scaled if is_linear else X_train.values
        X_v = X_val_scaled if is_linear else X_val.values
        
        try:
            model.fit(X_tr, y_train)
            val_preds = model.predict(X_v)
            val_preds = np.maximum(0.0, val_preds)
            metrics = compute_regression_metrics(y_val, val_preds)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).tolist()
            else:
                importances = None
                
            results[name] = {
                'model': model,
                'scaler': scaler if is_linear else None,
                'val_preds': val_preds,
                'metrics': metrics,
                'importances': importances
            }
        except (Exception, OSError) as e:
            print(f"[DEPENDENCY WARNING] {name} is unavailable or failed runtime execution: {e}. Skipping and continuing with remaining models.")
        
    return results

def evaluate_chained_energy(model, X_eval, y_eval, predicted_occupancy, is_linear=False, scaler=None):
    """
    Evaluates energy model under the Chained forecasting regime by substituting
    the ground-truth is_occupied_next_hour feature with the out-of-sample Stage 1 prediction.
    """
    X_chained = X_eval.copy()
    if 'is_occupied_next_hour' in X_chained.columns:
        X_chained['is_occupied_next_hour'] = np.asarray(predicted_occupancy, dtype=float)
        
    X_input = scaler.transform(X_chained) if (is_linear and scaler is not None) else X_chained.values
    preds = model.predict(X_input)
    preds = np.maximum(0.0, preds)
    metrics = compute_regression_metrics(y_eval, preds)
    
    return {
        'predictions': preds,
        'metrics': metrics
    }

def save_energy_model_artifacts(model, scaler, feature_names, metadata, save_dir):
    """
    Serializes champion energy model, preprocessor, feature list, and training metadata.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, out_dir / 'energy_regressor.joblib')
    if scaler is not None:
        joblib.dump(scaler, out_dir / 'energy_scaler.joblib')
        
    meta = {
        'feature_names': list(feature_names),
        'metadata': metadata
    }
    with open(out_dir / 'energy_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved energy model artifacts to: {out_dir}")

def load_energy_model_artifacts(save_dir):
    """
    Reloads saved energy regressor and metadata.
    """
    in_dir = Path(save_dir)
    model = joblib.load(in_dir / 'energy_regressor.joblib')
    scaler_path = in_dir / 'energy_scaler.joblib'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    with open(in_dir / 'energy_metadata.json', 'r') as f:
        meta = json.load(f)
        
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': meta['feature_names'],
        'metadata': meta['metadata']
    }
