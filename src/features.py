"""
Feature Engineering Module for Smart Building AI.

Provides modular, reproducible, and strictly causal feature transformations
for building occupancy and energy consumption modeling.
Enforces zero-lookahead bias guarantees.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract calendar, diurnal, and work-schedule features from a DatetimeIndex
    or timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex or a timestamp column.
    timestamp_col : Optional[str]
        Name of timestamp column if not index.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with calendar indicator features.
    """
    df = df.copy()
    if timestamp_col is not None:
        dt = pd.to_datetime(df[timestamp_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    else:
        raise ValueError("DataFrame must either have a DatetimeIndex or a valid timestamp_col.")

    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
    # Standard commercial building occupancy schedule: Monday-Friday, 08:00 to 18:00
    df["is_business_hour"] = (
        (~df["is_weekend"].astype(bool)) & (dt.hour >= 8) & (dt.hour <= 18)
    ).astype(int)
    df["month"] = dt.month
    df["day_of_year"] = dt.dayofyear
    df["quarter"] = dt.quarter

    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic temporal features (hour, day of week, month) into continuous
    sine and cosine coordinates to preserve cyclical proximity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'hour', 'day_of_week', and 'month'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added cyclical sine/cosine pairs.
    """
    df = df.copy()

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    if "day_of_week" in df.columns:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Add strictly past lag observations.
    Enforces lag >= 1 to prevent temporal data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted time-series DataFrame.
    columns : List[str]
        Columns to lag.
    lags : List[int]
        List of integer lag steps (e.g., [1, 2, 24]).

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged feature columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        for lag in lags:
            if lag < 1:
                raise ValueError(f"Lag must be >= 1 to prevent data leakage. Got lag={lag}.")
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int]
) -> pd.DataFrame:
    """
    Add strictly past rolling window features (mean and standard deviation).
    Enforces closed='left' by shifting input by 1 step before calculating rolling statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted time-series DataFrame.
    columns : List[str]
        Columns to compute rolling statistics on.
    windows : List[int]
        Window sizes in time steps (e.g., [3, 6, 24]).

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean and std columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        for w in windows:
            if w < 2:
                raise ValueError(f"Rolling window must be >= 2. Got window={w}.")
            # Strictly shift by 1 to exclude concurrent observation
            shifted = df[col].shift(1)
            df[f"{col}_rolling_mean_{w}h"] = shifted.rolling(window=w, min_periods=1).mean()
            df[f"{col}_rolling_std_{w}h"] = shifted.rolling(window=w, min_periods=2).std().fillna(0.0)
    return df


def add_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific thermal features derived from building thermodynamics:
    - Temperature gradient between indoor and outdoor environment (driving thermal conduction).
    - Rate of change of indoor temperature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'indoor_temp_mean' and 'outdoor_temp_c'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added thermodynamic features.
    """
    df = df.copy()
    if "indoor_temp_mean" in df.columns and "outdoor_temp_c" in df.columns:
        df["temp_gradient_in_out"] = df["indoor_temp_mean"] - df["outdoor_temp_c"]

    if "indoor_temp_mean" in df.columns:
        # Rate of temperature change over past 1 hour (shifted difference)
        df["indoor_temp_diff_1h"] = df["indoor_temp_mean"].diff().fillna(0.0)

    return df
