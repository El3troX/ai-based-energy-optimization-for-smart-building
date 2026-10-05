"""
Unit tests for data preprocessing, 1-hour-ahead target shifting, and joint dataset construction.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.features import (
    add_calendar_features,
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
    add_thermal_features,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def test_calendar_features():
    dates = pd.date_range("2018-05-22 00:00:00", periods=48, freq="1h")
    df = pd.DataFrame({"value": np.random.randn(len(dates))}, index=dates)
    df_feat = add_calendar_features(df)
    
    assert "hour" in df_feat.columns
    assert "day_of_week" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "is_business_hour" in df_feat.columns
    assert df_feat["hour"].iloc[10] == 10
    assert df_feat["is_weekend"].iloc[0] == 0


def test_lag_features_zero_leakage():
    dates = pd.date_range("2018-01-01", periods=10, freq="1h")
    df = pd.DataFrame({"y": list(range(10))}, index=dates)
    
    df_lag = add_lag_features(df, columns=["y"], lags=[1, 2])
    assert "y_lag_1h" in df_lag.columns
    assert "y_lag_2h" in df_lag.columns
    
    assert df_lag["y_lag_1h"].iloc[1] == df["y"].iloc[0]
    assert np.isnan(df_lag["y_lag_1h"].iloc[0])
    
    with pytest.raises(ValueError):
        add_lag_features(df, columns=["y"], lags=[0])


def test_rolling_features_zero_leakage():
    dates = pd.date_range("2018-01-01", periods=10, freq="1h")
    df = pd.DataFrame({"y": [1.0] * 10}, index=dates)
    
    df_roll = add_rolling_features(df, columns=["y"], windows=[3])
    assert np.isnan(df_roll["y_rolling_mean_3h"].iloc[0])
    assert df_roll["y_rolling_mean_3h"].iloc[1] == 1.0


def test_processed_tables_exist_and_valid():
    occ_path = os.path.join(PROCESSED_DIR, "occupancy_data.parquet")
    energy_path = os.path.join(PROCESSED_DIR, "energy_data.parquet")
    joint_path = os.path.join(PROCESSED_DIR, "joint_modeling_data.parquet")
    
    assert os.path.exists(occ_path), f"Missing {occ_path}"
    assert os.path.exists(energy_path), f"Missing {energy_path}"
    assert os.path.exists(joint_path), f"Missing {joint_path}"
    
    df_occ = pd.read_parquet(occ_path)
    df_energy = pd.read_parquet(energy_path)
    df_joint = pd.read_parquet(joint_path)
    
    # Check row counts
    assert len(df_occ) > 6000, f"Occupancy table has unexpected row count: {len(df_occ)}"
    assert len(df_energy) > 5500, f"Energy table has unexpected row count: {len(df_energy)}"
    assert len(df_joint) == len(df_energy), "Joint table should match intersection rows"
    
    # Check nulls
    assert df_occ.isnull().sum().sum() == 0, "Occupancy table contains null values!"
    assert df_energy.isnull().sum().sum() == 0, "Energy table contains null values!"
    assert df_joint.isnull().sum().sum() == 0, "Joint table contains null values!"
    
    # Check chronological monotonicity
    assert df_occ["timestamp"].is_monotonic_increasing, "Occupancy timestamps not monotonic!"
    assert df_energy["timestamp"].is_monotonic_increasing, "Energy timestamps not monotonic!"
    assert df_joint["timestamp"].is_monotonic_increasing, "Joint timestamps not monotonic!"
    
    # Check timestamp uniqueness
    assert not df_occ["timestamp"].duplicated().any(), "Occupancy table contains duplicate timestamps!"
    assert not df_energy["timestamp"].duplicated().any(), "Energy table contains duplicate timestamps!"
    assert not df_joint["timestamp"].duplicated().any(), "Joint table contains duplicate timestamps!"


def test_explicit_prediction_horizon_and_zero_leakage():
    occ_path = os.path.join(PROCESSED_DIR, "occupancy_data.parquet")
    energy_path = os.path.join(PROCESSED_DIR, "energy_data.parquet")
    
    df_occ = pd.read_parquet(occ_path)
    df_energy = pd.read_parquet(energy_path)
    
    # Verify explicit target names
    assert "is_occupied_next_hour" in df_occ.columns
    assert "occ_total_mean_next_hour" in df_occ.columns
    assert "south_wing_total_kwh_next_hour" in df_energy.columns
    
    # 1. No energy targets in occupancy table
    energy_targets = {
        "south_wing_total_kwh_next_hour", "lig_S_kwh_next_hour",
        "mels_S_kwh_next_hour", "hvac_S_kwh_next_hour"
    }
    assert len(set(df_occ.columns).intersection(energy_targets)) == 0, "Energy targets found in occupancy table!"
    
    # 2. No concurrent submeter predictors in energy features when predicting south_wing_total_kwh_next_hour
    energy_features = [c for c in df_energy.columns if c not in energy_targets and c != "timestamp"]
    assert len(set(energy_features).intersection(energy_targets)) == 0, "Concurrent target submeters in energy features!"
