"""
Train/validation splitting and forecast metrics for time series.
Time-based splits and MAE, RMSE, MAPE with safe handling of zeros.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def train_val_split(
    df: pd.DataFrame,
    date_col: str = "date",
    val_days_or_ratio: int | float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by time: train = earlier period, val = later period.

    Args:
        df: Must have date_col and be sortable by it.
        date_col: Column name containing dates.
        val_days_or_ratio: If int, number of most recent days for validation.
            If float in (0, 1), fraction of (unique) dates for validation.

    Returns:
        (train_df, val_df) with same columns as df.
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame columns: {list(df.columns)}")
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].drop_duplicates()
    dates = dates.sort_values().reset_index(drop=True)
    n = len(dates)
    if n == 0:
        raise ValueError("No dates found; DataFrame is empty or date_col has no values.")

    if isinstance(val_days_or_ratio, int):
        if val_days_or_ratio <= 0 or val_days_or_ratio >= n:
            raise ValueError(
                f"val_days_or_ratio as int must be in (0, n_dates); got {val_days_or_ratio}, n_dates={n}"
            )
        cutoff_idx = n - val_days_or_ratio
    else:
        if not 0 < val_days_or_ratio < 1:
            raise ValueError("val_days_or_ratio as float must be in (0, 1)")
        cutoff_idx = int(n * (1 - val_days_or_ratio))
        if cutoff_idx >= n or cutoff_idx < 1:
            raise ValueError(
                f"Split would leave train or val empty; n_dates={n}, ratio={val_days_or_ratio}"
            )

    cutoff_date = dates.iloc[cutoff_idx]
    train_df = df[df[date_col] < cutoff_date].copy()
    val_df = df[df[date_col] >= cutoff_date].copy()
    if train_df.empty or val_df.empty:
        raise AssertionError(
            f"Split produced empty set: train={len(train_df)}, val={len(val_df)}"
        )
    return train_df, val_df


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape; got {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape; got {y_true.shape} vs {y_pred.shape}")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    epsilon: float = 1e-8,
    skip_zeros: bool = True,
) -> float:
    """
    Mean absolute percentage error: mean(|y_true - y_pred| / |y_true|) * 100.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        epsilon: Lower bound for denominator when skip_zeros=False to avoid div-by-zero.
        skip_zeros: If True, rows where |y_true| < epsilon are excluded from the mean.
            If False, denominator is max(|y_true|, epsilon). Use skip_zeros=True for
            count data where zeros make MAPE misleading.

    Returns:
        MAPE as a percentage (0–100+).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape; got {y_true.shape} vs {y_pred.shape}")

    if skip_zeros:
        mask = np.abs(y_true) >= epsilon
        if not np.any(mask):
            raise ValueError(
                "All |y_true| values are below epsilon; MAPE undefined. Use skip_zeros=False or check data."
            )
        denom = np.abs(y_true[mask])
        num = np.abs(y_true[mask] - y_pred[mask])
    else:
        denom = np.maximum(np.abs(y_true), epsilon)
        num = np.abs(y_true - y_pred)
    pct = num / denom
    return float(np.mean(pct) * 100.0)
