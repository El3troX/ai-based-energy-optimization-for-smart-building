"""
Data Preprocessing and Harmonization Pipeline for Smart Building AI.

Processes the official Building 59 raw sensor streams, enforces sensor cleaning
rules (DS18B20 error codes, current transducer negative offsets, duplicate timestamps),
harmonizes multi-frequency channels into hourly physical units (kWh, °C, counts),
and constructs leakage-free modeling tables for occupancy and energy prediction.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from src.features import (
    add_calendar_features,
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
    add_thermal_features,
)


def load_clean_occupancy(
    filepath: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Load, clean, and resample camera ground-truth occupant counts.

    Parameters
    ----------
    filepath : str
        Path to raw 'occ.csv'.
    freq : str
        Resampling frequency (default '1h').

    Returns
    -------
    pd.DataFrame
        Cleaned hourly occupancy DataFrame.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"]).sort_values("timestamp")

    # Handle duplicates if any by taking the mean across duplicate timestamps
    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp").mean().reset_index()

    # Enforce non-negative counts
    for col in ["occ_third_south", "occ_fourth_south"]:
        df[col] = df[col].clip(lower=0.0)

    df["occ_total"] = df["occ_third_south"] + df["occ_fourth_south"]
    df = df.set_index("timestamp")

    # Resample to target frequency
    resampled = pd.DataFrame(index=df.resample(freq).asfreq().index)
    resampled["occ_total_mean"] = df["occ_total"].resample(freq).mean()
    resampled["occ_total_max"] = df["occ_total"].resample(freq).max()
    resampled["occ_third_south_mean"] = df["occ_third_south"].resample(freq).mean()
    resampled["occ_fourth_south_mean"] = df["occ_fourth_south"].resample(freq).mean()

    # Classification target: zone is occupied if max count during the hour > 0
    resampled["is_occupied"] = (resampled["occ_total_max"] > 0).astype(int)

    return resampled


def load_clean_electricity(
    filepath: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Load, clean, and resample electrical submeter data.
    Clamps negative transducer drift to 0.0.
    In an hourly framework, average active power (kW) multiplied by 1 hour equals
    electrical energy consumption in kilowatt-hours (kWh).

    Parameters
    ----------
    filepath : str
        Path to raw 'ele.csv'.
    freq : str
        Resampling frequency (default '1h').

    Returns
    -------
    pd.DataFrame
        Cleaned hourly electricity consumption in kWh.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"]).sort_values("timestamp")

    # Drop non-continuous 2020 channel if present
    if "Unnamed: 6" in df.columns:
        df = df.drop(columns=["Unnamed: 6"])

    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp").mean().reset_index()

    meter_cols = [c for c in df.columns if c != "timestamp"]
    for col in meter_cols:
        # Clamp sensor baseline zero-drift (e.g., -0.05 kW) to 0.0
        df[col] = df[col].clip(lower=0.0)

    df = df.set_index("timestamp")
    # Hourly resample: mean power (kW) * 1 hour = energy in kWh
    resampled = df.resample(freq).mean()

    # Linearly interpolate small telemetry dropouts (< 4 consecutive hours)
    resampled = resampled.interpolate(method="linear", limit=4)

    # Create explicit physical kWh targets for the South Wing
    out = pd.DataFrame(index=resampled.index)
    out["lig_S_kwh"] = resampled["lig_S"]
    out["mels_S_kwh"] = resampled["mels_S"]
    out["hvac_S_kwh"] = resampled["hvac_S"]
    out["south_wing_total_kwh"] = out["lig_S_kwh"] + out["mels_S_kwh"] + out["hvac_S_kwh"]

    # North Wing references for context
    if "mels_N" in resampled.columns and "hvac_N" in resampled.columns:
        out["mels_N_kwh"] = resampled["mels_N"]
        out["hvac_N_kwh"] = resampled["hvac_N"]

    return out


def load_clean_weather(
    filepath: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Load, clean, and resample on-site outdoor meteorological data.

    Parameters
    ----------
    filepath : str
        Path to raw 'site_weather.csv'.
    freq : str
        Resampling frequency (default '1h').

    Returns
    -------
    pd.DataFrame
        Cleaned hourly meteorological features.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"]).sort_values("timestamp")

    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp").mean().reset_index()

    df = df.set_index("timestamp")
    resampled = df.resample(freq).mean()

    out = pd.DataFrame(index=resampled.index)
    out["outdoor_temp_c"] = resampled["air_temp_set_1"]
    out["relative_humidity"] = resampled["relative_humidity_set_1"]
    out["dew_point_temp_c"] = resampled["dew_point_temperature_set_1d"]
    out["solar_radiation"] = resampled["solar_radiation_set_1"].clip(lower=0.0)

    return out


def load_clean_indoor_temperature(
    filepath: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Load, clean, and resample 16 indoor zone temperature loggers.
    Treats known DS18B20 hardware reset artifacts (85.0°C and 0.0°C) as sensor faults,
    replaces them with NaN, and interpolates before hourly spatial aggregation.

    Parameters
    ----------
    filepath : str
        Path to raw 'zone_temp_interior.csv'.
    freq : str
        Resampling frequency (default '1h').

    Returns
    -------
    pd.DataFrame
        Cleaned hourly indoor temperature statistics (°C).
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"]).sort_values("timestamp")

    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp").mean().reset_index()

    logger_cols = [c for c in df.columns if c != "timestamp"]
    for col in logger_cols:
        # DS18B20 power-on reset code is exactly 85.0°C; disconnection code is 0.0°C
        # Plausible indoor temperature in research office is 15°C to 35°C
        invalid_mask = (df[col] == 85.0) | (df[col] == 0.0) | (df[col] < 12.0) | (df[col] > 38.0)
        df.loc[invalid_mask, col] = np.nan
        # Interpolate across short transient gaps
        df[col] = df[col].interpolate(method="linear", limit=12)

    df = df.set_index("timestamp")
    resampled_loggers = df[logger_cols].resample(freq).mean()

    out = pd.DataFrame(index=resampled_loggers.index)
    out["indoor_temp_mean"] = resampled_loggers.mean(axis=1)
    out["indoor_temp_min"] = resampled_loggers.min(axis=1)
    out["indoor_temp_max"] = resampled_loggers.max(axis=1)

    # Fill residual logger dropout edge gaps
    out["indoor_temp_mean"] = out["indoor_temp_mean"].interpolate(method="linear", limit=4)
    out["indoor_temp_min"] = out["indoor_temp_min"].interpolate(method="linear", limit=4)
    out["indoor_temp_max"] = out["indoor_temp_max"].interpolate(method="linear", limit=4)

    return out


def load_clean_rtu_south(
    clean_dir: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Load, clean, and resample South Wing Rooftop Units (RTU 003 & RTU 004).

    Parameters
    ----------
    clean_dir : str
        Directory containing clean CSV files.
    freq : str
        Resampling frequency (default '1h').

    Returns
    -------
    pd.DataFrame
        Cleaned hourly South Wing HVAC operational features.
    """
    def _read_and_dedup(filename: str, cols: list) -> pd.DataFrame:
        p = os.path.join(clean_dir, filename)
        d = pd.read_csv(p, usecols=["date"] + cols)
        d["timestamp"] = pd.to_datetime(d["date"])
        d = d.drop(columns=["date"]).sort_values("timestamp")
        if d["timestamp"].duplicated().any():
            d = d.groupby("timestamp").mean().reset_index()
        return d.set_index("timestamp")[cols].resample(freq).mean()

    # RTU 003 & 004 Supply Fan Speeds (%)
    df_fan = _read_and_dedup("rtu_fan_spd.csv", ["rtu_003_sf_vfd_spd_fbk_tn", "rtu_004_sf_vfd_spd_fbk_tn"])
    # RTU 003 & 004 Outdoor Air Damper Positions (%)
    df_damp = _read_and_dedup("rtu_oa_damper.csv", ["rtu_003_oadmpr_pct", "rtu_004_oadmpr_pct"])

    out = pd.DataFrame(index=df_fan.index)
    out["rtu_south_fan_spd_mean"] = df_fan.mean(axis=1)
    out["rtu_south_damper_pct_mean"] = df_damp.mean(axis=1)

    return out


def build_modeling_tables(
    raw_dir: str,
    processed_dir: str,
    start_time: str = "2018-05-22 07:00:00",
    end_time: str = "2019-02-21 10:00:00"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Construct standardized, leakage-free modeling tables for occupancy and energy prediction,
    constrained to the verified empirical sensor overlap period.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw dataset files.
    processed_dir : str
        Directory where processed Parquet/CSV files will be saved.
    start_time : str
        Overlap start boundary.
    end_time : str
        Overlap end boundary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (occupancy_df, energy_df, validation_metrics)
    """
    clean_dir = os.path.join(raw_dir, "Bldg59_clean data")
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load and resample all verified sources to 1-hour
    occ_h = load_clean_occupancy(os.path.join(clean_dir, "occ.csv"), freq="1h")
    ele_h = load_clean_electricity(os.path.join(clean_dir, "ele.csv"), freq="1h")
    wea_h = load_clean_weather(os.path.join(clean_dir, "site_weather.csv"), freq="1h")
    zt_h = load_clean_indoor_temperature(os.path.join(clean_dir, "zone_temp_interior.csv"), freq="1h")
    rtu_h = load_clean_rtu_south(clean_dir, freq="1h")

    # 2. Join across shared timestamp index
    combined = pd.concat([occ_h, ele_h, wea_h, zt_h, rtu_h], axis=1, join="inner")

    # 3. Restrict strictly to verified overlap window
    t_start = pd.to_datetime(start_time)
    t_end = pd.to_datetime(end_time)
    combined = combined.loc[(combined.index >= t_start) & (combined.index <= t_end)].copy()
    initial_rows = len(combined)

    # 4. Feature Engineering: Calendar and Thermodynamic features
    combined = add_calendar_features(combined)
    combined = add_cyclical_time_features(combined)
    combined = add_thermal_features(combined)

    # 5. Build OCCUPANCY MODELING TABLE
    # Targets: is_occupied (binary), occ_total_mean (continuous headcount)
    # Features: Environmental, weather, HVAC status, temporal features, and strictly causal lagged occupancy.
    # Excludes concurrent electrical consumption to prevent target leakage.
    occ_df = combined.copy()
    occ_df = add_lag_features(occ_df, columns=["occ_total_mean", "is_occupied"], lags=[1, 2, 24])
    occ_df = add_rolling_features(occ_df, columns=["occ_total_mean"], windows=[3, 6, 24])

    occ_feature_cols = [
        # Environmental
        "indoor_temp_mean", "indoor_temp_min", "indoor_temp_max",
        "indoor_temp_diff_1h", "temp_gradient_in_out",
        # Meteorological
        "outdoor_temp_c", "relative_humidity", "dew_point_temp_c", "solar_radiation",
        # HVAC operational status
        "rtu_south_fan_spd_mean", "rtu_south_damper_pct_mean",
        # Temporal / Calendar
        "hour", "day_of_week", "is_weekend", "is_business_hour", "month",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        # Causal Lags & Rolling
        "occ_total_mean_lag_1h", "occ_total_mean_lag_2h", "occ_total_mean_lag_24h",
        "is_occupied_lag_1h", "is_occupied_lag_2h", "is_occupied_lag_24h",
        "occ_total_mean_rolling_mean_3h", "occ_total_mean_rolling_std_3h",
        "occ_total_mean_rolling_mean_6h", "occ_total_mean_rolling_std_6h",
        "occ_total_mean_rolling_mean_24h", "occ_total_mean_rolling_std_24h",
    ]
    occ_target_cols = ["is_occupied", "occ_total_mean"]

    occupancy_data = occ_df[occ_target_cols + occ_feature_cols].copy()
    # Drop rows where lag is unavailable
    occupancy_data = occupancy_data.dropna().copy()
    occupancy_data.reset_index(inplace=True)

    # 6. Build ENERGY MODELING TABLE
    # Targets: south_wing_total_kwh, lig_S_kwh, mels_S_kwh, hvac_S_kwh
    # Features: Occupancy state, environmental, weather, HVAC status, temporal features,
    # strictly causal lagged energy. Excludes concurrent submeter predictors when predicting total.
    energy_df = combined.copy()
    energy_df = add_lag_features(
        energy_df,
        columns=["south_wing_total_kwh", "lig_S_kwh", "mels_S_kwh", "hvac_S_kwh"],
        lags=[1, 2, 24]
    )
    energy_df = add_rolling_features(
        energy_df,
        columns=["south_wing_total_kwh"],
        windows=[3, 6, 24]
    )

    energy_feature_cols = [
        # Occupancy Predictors
        "is_occupied", "occ_total_mean",
        # Environmental & Meteorological
        "indoor_temp_mean", "indoor_temp_diff_1h", "temp_gradient_in_out",
        "outdoor_temp_c", "relative_humidity", "dew_point_temp_c", "solar_radiation",
        # HVAC Controllable Operations
        "rtu_south_fan_spd_mean", "rtu_south_damper_pct_mean",
        # Temporal / Calendar
        "hour", "day_of_week", "is_weekend", "is_business_hour", "month",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        # Strictly Causal Past Energy Lags & Rolling
        "south_wing_total_kwh_lag_1h", "south_wing_total_kwh_lag_2h", "south_wing_total_kwh_lag_24h",
        "lig_S_kwh_lag_1h", "lig_S_kwh_lag_2h", "lig_S_kwh_lag_24h",
        "mels_S_kwh_lag_1h", "mels_S_kwh_lag_2h", "mels_S_kwh_lag_24h",
        "hvac_S_kwh_lag_1h", "hvac_S_kwh_lag_2h", "hvac_S_kwh_lag_24h",
        "south_wing_total_kwh_rolling_mean_3h", "south_wing_total_kwh_rolling_std_3h",
        "south_wing_total_kwh_rolling_mean_6h", "south_wing_total_kwh_rolling_std_6h",
        "south_wing_total_kwh_rolling_mean_24h", "south_wing_total_kwh_rolling_std_24h",
    ]
    energy_target_cols = ["south_wing_total_kwh", "lig_S_kwh", "mels_S_kwh", "hvac_S_kwh"]

    energy_data = energy_df[energy_target_cols + energy_feature_cols].copy()
    energy_data = energy_data.dropna().copy()
    energy_data.reset_index(inplace=True)

    # 7. Integrity Verifications
    for name, df_table in [("occupancy_data", occupancy_data), ("energy_data", energy_data)]:
        assert df_table["timestamp"].is_monotonic_increasing, f"{name} is not chronologically sorted!"
        assert not df_table["timestamp"].duplicated().any(), f"{name} contains duplicate timestamps!"
        assert df_table.isnull().sum().sum() == 0, f"{name} contains unhandled NaN values!"

    # 8. Export to Parquet and CSV
    occ_parquet_path = os.path.join(processed_dir, "occupancy_data.parquet")
    occ_csv_path = os.path.join(processed_dir, "occupancy_data.csv")
    occupancy_data.to_parquet(occ_parquet_path, index=False)
    occupancy_data.to_csv(occ_csv_path, index=False)

    energy_parquet_path = os.path.join(processed_dir, "energy_data.parquet")
    energy_csv_path = os.path.join(processed_dir, "energy_data.csv")
    energy_data.to_parquet(energy_parquet_path, index=False)
    energy_data.to_csv(energy_csv_path, index=False)

    # 9. Validation Report Summary
    validation_report = {
        "benchmark_window": f"{start_time} to {end_time}",
        "initial_aligned_rows": initial_rows,
        "occupancy_table": {
            "rows": len(occupancy_data),
            "columns": len(occupancy_data.columns),
            "start_time": str(occupancy_data["timestamp"].min()),
            "end_time": str(occupancy_data["timestamp"].max()),
            "targets": occ_target_cols,
            "features_count": len(occ_feature_cols),
            "features": occ_feature_cols,
            "missingness": int(occupancy_data.isnull().sum().sum()),
            "rows_removed_due_to_lags": initial_rows - len(occupancy_data),
            "percentage_removed": round((initial_rows - len(occupancy_data)) / initial_rows * 100, 2),
            "is_monotonic": bool(occupancy_data["timestamp"].is_monotonic_increasing),
            "is_unique_timestamps": bool(not occupancy_data["timestamp"].duplicated().any())
        },
        "energy_table": {
            "rows": len(energy_data),
            "columns": len(energy_data.columns),
            "start_time": str(energy_data["timestamp"].min()),
            "end_time": str(energy_data["timestamp"].max()),
            "targets": energy_target_cols,
            "features_count": len(energy_feature_cols),
            "features": energy_feature_cols,
            "missingness": int(energy_data.isnull().sum().sum()),
            "rows_removed_due_to_lags": initial_rows - len(energy_data),
            "percentage_removed": round((initial_rows - len(energy_data)) / initial_rows * 100, 2),
            "is_monotonic": bool(energy_data["timestamp"].is_monotonic_increasing),
            "is_unique_timestamps": bool(not energy_data["timestamp"].duplicated().any())
        }
    }

    return occupancy_data, energy_data, validation_report
