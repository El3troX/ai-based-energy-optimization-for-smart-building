"""
Model Evaluation and Diagnostic Visualization Suite.
Provides rigorous metric computation and academic plotting routines for:
- Binary occupancy classification (F1, PR-AUC, ROC-AUC, Balanced Accuracy)
- Headcount & Energy continuous regression (MAE, RMSE, R^2, MAPE)
- Optimal classification threshold tuning on validation splits
- Error propagation analysis for chained BEMS forecasting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes comprehensive classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'confusion_matrix': cm
    }
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics['roc_auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics['roc_auc'] = None
        try:
            metrics['pr_auc'] = round(float(average_precision_score(y_true, y_prob)), 4)
        except Exception:
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
        
    return metrics

def compute_regression_metrics(y_true, y_pred):
    """
    Computes regression performance metrics: MAE, RMSE, R^2, and MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    # MAPE with numerical safety epsilon
    non_zero_mask = np.abs(y_true) > 1e-3
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = None
        
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'mape': round(mape, 2) if mape is not None else None
    }

def tune_classification_threshold(y_true, y_prob, metric='f1', step=0.01):
    """
    Evaluates classification thresholds on validation predictions to maximize a chosen metric.
    Prevents data leakage by ensuring tuning is restricted to the validation split.
    """
    thresholds = np.arange(0.05, 0.95 + step, step)
    scores = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        scores.append(score)
        
    best_idx = int(np.argmax(scores))
    best_threshold = float(round(thresholds[best_idx], 3))
    best_score = float(round(scores[best_idx], 4))
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'metric': metric,
        'thresholds': thresholds.tolist(),
        'scores': scores
    }

# --- Plotting Functions ---

def plot_confusion_matrix(cm, labels=['Unoccupied (0)', 'Occupied (1)'], title='Confusion Matrix', save_path=None):
    """Plots a formatted confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_pr_curves(y_true, y_prob, title_prefix='Occupancy Classifier', save_path=None):
    """Plots ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # ROC Curve
    ax = axes[0]
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Chance')
    ax.set_title(f'{title_prefix} — ROC Curve', fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    
    # PR Curve
    ax = axes[1]
    ax.plot(rec, prec, color='#10b981', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    baseline = np.mean(y_true)
    ax.axhline(baseline, color='#9ca3af', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_title(f'{title_prefix} — Precision-Recall Curve', fontsize=11, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_actual_vs_predicted_series(y_true, y_pred, index=None, title='Actual vs Predicted Energy', save_path=None, sample_hours=24*14):
    """Plots actual vs predicted time series and scatter correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    
    # Time-series snippet
    ax = axes[0]
    plot_idx = index[:sample_hours] if index is not None else np.arange(min(sample_hours, len(y_true)))
    ax.plot(plot_idx, y_true[:sample_hours], label='Actual', color='#111827', lw=1.5)
    ax.plot(plot_idx, y_pred[:sample_hours], label='Predicted', color='#2563eb', lw=1.5, alpha=0.85)
    ax.set_title(f'{title} (Time Series)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Energy (kWh)')
    ax.legend()
    if index is not None:
        ax.tick_params(axis='x', rotation=30)
        
    # Scatter plot
    ax = axes[1]
    ax.scatter(y_true, y_pred, alpha=0.3, color='#2563eb', s=12)
    min_v, max_v = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='1:1 Ideal Identity')
    ax.set_title(f'{title} (Scatter)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Energy (kWh)')
    ax.set_ylabel('Predicted Energy (kWh)')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, index=None, title='Energy Forecast Residuals', save_path=None):
    """Plots residual distribution and residual timeline."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    
    # Histogram & KDE
    ax = axes[0]
    sns.histplot(residuals, bins=40, kde=True, color='#4b5563', ax=ax)
    ax.axvline(0, color='r', linestyle='--', lw=1.5)
    ax.set_title(f'{title} — Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    
    # Residuals over time
    ax = axes[1]
    plot_idx = index if index is not None else np.arange(len(residuals))
    ax.plot(plot_idx, residuals, color='#2563eb', lw=0.8, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', lw=1.5)
    ax.set_title(f'{title} — Timeline', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residual (kWh)')
    if index is not None:
        ax.tick_params(axis='x', rotation=30)
        
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_feature_importance(feature_names, importances, top_n=15, title='Feature Importance', save_path=None):
    """Plots horizontal bar chart of top feature importances."""
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='Blues_r', ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_oracle_vs_chained(y_true, y_oracle, y_chained, index=None, title='Oracle vs Chained Forecast Comparison', save_path=None, sample_hours=24*14):
    """Compares oracle energy predictions (true occupancy) vs chained energy predictions (predicted occupancy)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    plot_idx = index[:sample_hours] if index is not None else np.arange(min(sample_hours, len(y_true)))
    
    # Top: Trajectories
    ax = axes[0]
    ax.plot(plot_idx, y_true[:sample_hours], label='Actual Energy', color='#111827', lw=1.8)
    ax.plot(plot_idx, y_oracle[:sample_hours], label='Oracle Energy (Ground-Truth Occ)', color='#10b981', lw=1.4, linestyle='--')
    ax.plot(plot_idx, y_chained[:sample_hours], label='Chained Energy (Predicted Occ)', color='#ef4444', lw=1.4, alpha=0.9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy (kWh)')
    ax.legend(loc='upper right')
    
    # Bottom: Absolute Error Discrepancy
    ax = axes[1]
    oracle_err = np.abs(y_true[:sample_hours] - y_oracle[:sample_hours])
    chained_err = np.abs(y_true[:sample_hours] - y_chained[:sample_hours])
    err_delta = chained_err - oracle_err
    
    ax.bar(plot_idx, err_delta, width=0.03 if index is not None else 0.8, 
           color=np.where(err_delta >= 0, '#ef4444', '#10b981'), alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_title('Error Propagation Penalty (|Chained Error| - |Oracle Error|)', fontsize=11)
    ax.set_ylabel('Error Difference (kWh)')
    if index is not None:
        ax.tick_params(axis='x', rotation=30)
        
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
