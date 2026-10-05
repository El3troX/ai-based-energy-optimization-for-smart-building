"""
Stage 1: Occupancy Forecasting Models (t -> t+1).
Implements binary classification (is_occupied_next_hour) and continuous headcount
regression (occ_total_mean_next_hour) using strictly historical predictors (no lookahead,
no energy features in the predictor matrix).
"""
import os
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    tune_classification_threshold
)

OCC_EXCLUDED_COLS = [
    'timestamp',
    'is_occupied_next_hour',
    'occ_total_mean_next_hour',
    # Ensure zero energy submeter contamination in occupancy feature space
    'south_wing_total_kwh_next_hour',
    'lig_S_kwh_next_hour',
    'mels_S_kwh_next_hour',
    'hvac_S_kwh_next_hour',
    'south_wing_total_kwh_lag_1h',
    'south_wing_total_kwh_lag_2h',
    'south_wing_total_kwh_lag_24h',
    'lig_S_kwh_lag_1h',
    'lig_S_kwh_lag_2h',
    'lig_S_kwh_lag_24h',
    'mels_S_kwh_lag_1h',
    'mels_S_kwh_lag_2h',
    'mels_S_kwh_lag_24h',
    'hvac_S_kwh_lag_1h',
    'hvac_S_kwh_lag_2h',
    'hvac_S_kwh_lag_24h',
    'south_wing_total_kwh_rolling_mean_3h',
    'south_wing_total_kwh_rolling_std_3h',
    'south_wing_total_kwh_rolling_mean_6h',
    'south_wing_total_kwh_rolling_std_6h',
    'south_wing_total_kwh_rolling_mean_24h',
    'south_wing_total_kwh_rolling_std_24h',
    'rtu_south_fan_spd_mean_next_hour',
    'rtu_south_damper_pct_mean_next_hour'
]

def load_and_split_occupancy_data(data_path):
    """
    Loads joint or occupancy parquet data and applies the frozen chronological split.
    """
    df = pd.read_parquet(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
    train_mask = (df.index >= '2018-05-23 07:00:00') & (df.index <= '2018-11-30 23:00:00')
    val_mask = (df.index >= '2018-12-01 00:00:00') & (df.index <= '2019-01-10 23:00:00')
    test_mask = (df.index >= '2019-01-11 00:00:00') & (df.index <= '2019-02-21 09:00:00')
    
    feature_cols = [c for c in df.columns if c not in OCC_EXCLUDED_COLS]
    
    X_train = df.loc[train_mask, feature_cols].copy()
    y_train_cls = df.loc[train_mask, 'is_occupied_next_hour'].astype(int).values
    y_train_reg = df.loc[train_mask, 'occ_total_mean_next_hour'].values
    
    X_val = df.loc[val_mask, feature_cols].copy()
    y_val_cls = df.loc[val_mask, 'is_occupied_next_hour'].astype(int).values
    y_val_reg = df.loc[val_mask, 'occ_total_mean_next_hour'].values
    
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test_cls = df.loc[test_mask, 'is_occupied_next_hour'].astype(int).values
    y_test_reg = df.loc[test_mask, 'occ_total_mean_next_hour'].values
    
    return {
        'feature_cols': feature_cols,
        'train': {'X': X_train, 'y_cls': y_train_cls, 'y_reg': y_train_reg, 'index': X_train.index},
        'val': {'X': X_val, 'y_cls': y_val_cls, 'y_reg': y_val_reg, 'index': X_val.index},
        'test': {'X': X_test, 'y_cls': y_test_cls, 'y_reg': y_test_reg, 'index': X_test.index}
    }

def get_candidate_classifiers(random_state=42):
    """
    Returns a dictionary of candidate classification algorithms for is_occupied_next_hour.
    """
    return {
        'Logistic Regression (L2)': LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1)
    }

def get_candidate_headcount_regressors(random_state=42):
    """
    Returns candidate regression models for occ_total_mean_next_hour.
    """
    return {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbose=-1)
    }

def train_and_eval_classifiers(X_train, y_train, X_val, y_val, feature_cols, random_state=42):
    """
    Fits all candidate classifiers on Train, evaluates on Validation at default (0.5)
    and tuned thresholds, and identifies the validation champion.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = get_candidate_classifiers(random_state=random_state)
    results = {}
    
    for name, model in models.items():
        is_linear = 'Logistic' in name
        X_tr = X_train_scaled if is_linear else X_train.values
        X_v = X_val_scaled if is_linear else X_val.values
        
        try:
            model.fit(X_tr, y_train)
            
            # Probabilities on validation split
            val_probs = model.predict_proba(X_v)[:, 1]
            
            # Metrics at default threshold 0.5
            val_preds_default = (val_probs >= 0.5).astype(int)
            metrics_default = compute_classification_metrics(y_val, val_preds_default, val_probs)
            
            # Optimal threshold search on validation set (maximizing F1)
            tuning_info = tune_classification_threshold(y_val, val_probs, metric='f1')
            best_t = tuning_info['best_threshold']
            val_preds_tuned = (val_probs >= best_t).astype(int)
            metrics_tuned = compute_classification_metrics(y_val, val_preds_tuned, val_probs)
            
            # Extract feature importances if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]).tolist()
            else:
                importances = None
                
            results[name] = {
                'model': model,
                'scaler': scaler if is_linear else None,
                'val_probs': val_probs,
                'val_preds_default': val_preds_default,
                'val_preds_tuned': val_preds_tuned,
                'best_threshold': best_t,
                'metrics_default': metrics_default,
                'metrics_tuned': metrics_tuned,
                'importances': importances
            }
        except (Exception, OSError) as e:
            print(f"[DEPENDENCY WARNING] {name} is unavailable or failed runtime execution: {e}. Skipping and continuing with remaining models.")
        
    return results

def train_and_eval_headcount_regressors(X_train, y_train, X_val, y_val, random_state=42):
    """
    Trains headcount regression models on raw and log1p targets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = get_candidate_headcount_regressors(random_state=random_state)
    results = {}
    
    for name, model in models.items():
        is_linear = 'Ridge' in name
        X_tr = X_train_scaled if is_linear else X_train.values
        X_v = X_val_scaled if is_linear else X_val.values
        
        try:
            # Raw Target Training
            model.fit(X_tr, y_train)
            raw_val_pred = np.maximum(0.0, model.predict(X_v))
            raw_metrics = compute_regression_metrics(y_val, raw_val_pred)
            
            # Log1p Target Training
            y_train_log = np.log1p(np.maximum(0.0, y_train))
            model.fit(X_tr, y_train_log)
            log_val_pred = np.expm1(model.predict(X_v))
            log_val_pred = np.maximum(0.0, log_val_pred)
            log_metrics = compute_regression_metrics(y_val, log_val_pred)
            
            results[name] = {
                'raw_metrics': raw_metrics,
                'log_metrics': log_metrics,
                'preferred_regime': 'raw' if raw_metrics['rmse'] <= log_metrics['rmse'] else 'log1p'
            }
        except (Exception, OSError) as e:
            print(f"[DEPENDENCY WARNING] {name} is unavailable or failed runtime execution: {e}. Skipping and continuing with remaining models.")
        
    return results

def save_occupancy_model_artifacts(model, scaler, feature_names, threshold, metadata, save_dir):
    """
    Serializes champion occupancy model, preprocessing scaler, feature list, and metadata.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, out_dir / 'occupancy_classifier.joblib')
    if scaler is not None:
        joblib.dump(scaler, out_dir / 'occupancy_scaler.joblib')
        
    meta = {
        'threshold': float(threshold),
        'feature_names': list(feature_names),
        'metadata': metadata
    }
    with open(out_dir / 'occupancy_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved occupancy model artifacts to: {out_dir}")

def load_occupancy_model_artifacts(save_dir):
    """
    Reloads saved occupancy classifier and metadata.
    """
    in_dir = Path(save_dir)
    model = joblib.load(in_dir / 'occupancy_classifier.joblib')
    scaler_path = in_dir / 'occupancy_scaler.joblib'
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    with open(in_dir / 'occupancy_metadata.json', 'r') as f:
        meta = json.load(f)
        
    return {
        'model': model,
        'scaler': scaler,
        'threshold': meta['threshold'],
        'feature_names': meta['feature_names'],
        'metadata': meta['metadata']
    }
