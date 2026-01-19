# mco/data_validation.py
"""
Data quality validation utilities for MCO pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_customer_daily(df: pd.DataFrame, min_days: int = 30) -> Optional[str]:
    """
    Validate customer_daily DataFrame before feature engineering.
    Returns error message if validation fails, None if OK.
    """
    required_cols = ["customer_id", "as_of_date", "base_balance_eod", "netflow_sgd", "inflow_sgd", "outflow_sgd"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return f"Missing columns: {missing_cols}"

    if len(df) == 0:
        return "customer_daily is empty"

    n_customers = df["customer_id"].nunique()
    if n_customers == 0:
        return "No valid customer IDs"

    # Check date continuity per customer
    avg_days_per_customer = len(df) / n_customers
    if avg_days_per_customer < min_days:
        return f"Insufficient history: avg {avg_days_per_customer:.1f} days/customer (need {min_days})"

    # Check for extreme outliers
    balance_range = df["base_balance_eod"].max() - df["base_balance_eod"].min()
    if balance_range > 1e12:
        logger.warning(f"Very large balance range detected: {balance_range:.2e}")

    nans_total = df.isna().sum().sum()
    if nans_total > 0:
        logger.warning(f"customer_daily has {nans_total} NaN values")

    return None


def validate_labels(df: pd.DataFrame) -> Optional[str]:
    """
    Validate labels DataFrame before model training.
    Returns error message if validation fails, None if OK.
    """
    required_cols = ["customer_id", "obs_date", "horizon_bucket", "mco_amount_sgd"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return f"Missing columns: {missing_cols}"

    if len(df) == 0:
        return "labels is empty"

    # Check for NaNs in critical columns
    nans_mco = df["mco_amount_sgd"].isna().sum()
    if nans_mco > 0:
        logger.warning(f"Found {nans_mco} NaN MCO values in labels")

    # Check for negative MCO (should not happen by definition)
    neg_mco = (df["mco_amount_sgd"] < -1e-6).sum()
    if neg_mco > 0:
        logger.warning(f"Found {neg_mco} negative MCO values (expected non-negative)")

    return None


def validate_features(df: pd.DataFrame, min_coverage: float = 0.8) -> Optional[str]:
    """
    Validate features DataFrame before model training.
    min_coverage: minimum fraction of non-NaN values per column (default 80%)
    """
    if len(df) == 0:
        return "features is empty"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return "No numeric feature columns found"

    # Check NaN coverage
    coverage = 1.0 - (df[numeric_cols].isna().sum() / len(df))
    low_coverage = coverage[coverage < min_coverage]
    if len(low_coverage) > 0:
        logger.warning(f"Low coverage features (<{min_coverage*100:.0f}%): {list(low_coverage.index)}")
        if (coverage < 0.5).any():
            return f"Critical features with <50% coverage: {list(coverage[coverage < 0.5].index)}"

    return None


def profile_data(
    customer_daily: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame
) -> dict:
    """
    Generate data profiling summary for logging/reporting.
    """
    return {
        "customer_daily": {
            "rows": len(customer_daily),
            "customers": customer_daily["customer_id"].nunique(),
            "date_range": {
                "min": str(customer_daily["as_of_date"].min().date()),
                "max": str(customer_daily["as_of_date"].max().date()),
            },
            "balance_range": {
                "min": float(customer_daily["base_balance_eod"].min()),
                "max": float(customer_daily["base_balance_eod"].max()),
                "mean": float(customer_daily["base_balance_eod"].mean()),
            },
        },
        "labels": {
            "rows": len(labels),
            "horizons": labels["horizon_bucket"].nunique(),
            "mco_range": {
                "min": float(labels["mco_amount_sgd"].min()),
                "max": float(labels["mco_amount_sgd"].max()),
                "mean": float(labels["mco_amount_sgd"].mean()),
            },
            "coverage": float(1.0 - labels["mco_amount_sgd"].isna().sum() / len(labels)),
        },
        "features": {
            "rows": len(features),
            "columns": len(features.columns),
            "nan_count": int(features.isna().sum().sum()),
        },
    }
