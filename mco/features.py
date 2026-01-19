# mco/features.py
"""
Feature building (no leakage):
- rolling 30d outflow/balance stats
- salary recency
- volatility/tail signals
"""

from __future__ import annotations

import logging
import pandas as pd

from .config import ROLLING_WINDOW_DAYS, ROLLING_MIN_PERIODS, DEFAULT_DAYS_SINCE_SALARY_MISSING

logger = logging.getLogger(__name__)


def build_features(customer_daily: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    df = customer_daily.sort_values(["customer_id", "as_of_date"]).copy() if "entity_code" not in customer_daily.columns else customer_daily.sort_values(["entity_code", "customer_id", "as_of_date"]).copy()
    
    has_entity = "entity_code" in df.columns

    # ---- Salary last date per customer (per entity if multi-entity) ----
    txn2 = txn.copy()
    txn2["txn_date"] = txn2["txn_datetime"].dt.normalize()
    txn2 = txn2[
        (txn2["txn_type_code"] == "SALARY_CREDIT") &
        (txn2["dr_cr_flag"].astype(str).str.upper() == "CR")
    ]
    if has_entity and "entity_code" in txn2.columns:
        last_sal = txn2.groupby(["entity_code", "customer_id"])["txn_date"].max() if len(txn2) else pd.Series(dtype="datetime64[ns]")
    else:
        last_sal = txn2.groupby("customer_id")["txn_date"].max() if len(txn2) else pd.Series(dtype="datetime64[ns]")

    window = int(ROLLING_WINDOW_DAYS)
    minp = int(ROLLING_MIN_PERIODS)

    frames = []
    if has_entity:
        for entity, entity_group in df.groupby("entity_code", sort=False):
            for cid, g in entity_group.groupby("customer_id", sort=False):
                g = g.sort_values("as_of_date").copy()

                # Outflow / netflow stats (use rolling for all, expanding fills NaN early rows)
                g["avg_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).mean()
                # For early observations with NaN, use expanding window
                early_nan = g["avg_outflow_30"].isna()
                if early_nan.any():
                    g.loc[early_nan, "avg_outflow_30"] = g["outflow_sgd"].expanding(min_periods=minp).mean().loc[early_nan]
                
                g["p95_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).quantile(0.95)
                g["outflow_days_30"] = (g["outflow_sgd"] > 0).rolling(window, min_periods=minp).sum()
                g["netflow_mean_30"] = g["netflow_sgd"].rolling(window, min_periods=minp).mean()

                # Tail/volatility signals
                g["outflow_std_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).std()
                g["max_single_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).max()

                # Balance stats
                g["avg_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).mean()
                early_nan_bal = g["avg_bal_30"].isna()
                if early_nan_bal.any():
                    g.loc[early_nan_bal, "avg_bal_30"] = g["base_balance_eod"].expanding(min_periods=minp).mean().loc[early_nan_bal]
                
                g["std_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).std()
                g["min_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).min()

                # Salary recency
                try:
                    lsd = last_sal.get((entity, cid), pd.NaT)
                except:
                    lsd = pd.NaT
                if pd.isna(lsd):
                    g["days_since_salary"] = float(DEFAULT_DAYS_SINCE_SALARY_MISSING)
                else:
                    g["days_since_salary"] = (g["as_of_date"] - pd.Timestamp(lsd)).dt.days.clip(0, DEFAULT_DAYS_SINCE_SALARY_MISSING).astype(float)

                frames.append(g)
    else:
        for cid, g in df.groupby("customer_id", sort=False):
            g = g.sort_values("as_of_date").copy()

            # Outflow / netflow stats (use rolling for all, expanding fills NaN early rows)
            g["avg_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).mean()
            # For early observations with NaN, use expanding window
            early_nan = g["avg_outflow_30"].isna()
            if early_nan.any():
                g.loc[early_nan, "avg_outflow_30"] = g["outflow_sgd"].expanding(min_periods=minp).mean().loc[early_nan]
            
            g["p95_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).quantile(0.95)
            g["outflow_days_30"] = (g["outflow_sgd"] > 0).rolling(window, min_periods=minp).sum()
            g["netflow_mean_30"] = g["netflow_sgd"].rolling(window, min_periods=minp).mean()

            # Tail/volatility signals
            g["outflow_std_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).std()
            g["max_single_outflow_30"] = g["outflow_sgd"].rolling(window, min_periods=minp).max()

            # Balance stats
            g["avg_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).mean()
            early_nan_bal = g["avg_bal_30"].isna()
            if early_nan_bal.any():
                g.loc[early_nan_bal, "avg_bal_30"] = g["base_balance_eod"].expanding(min_periods=minp).mean().loc[early_nan_bal]
            
            g["std_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).std()
            g["min_bal_30"] = g["base_balance_eod"].rolling(window, min_periods=minp).min()

            # Salary recency
            lsd = last_sal.get(cid, pd.NaT)
            if pd.isna(lsd):
                g["days_since_salary"] = float(DEFAULT_DAYS_SINCE_SALARY_MISSING)
            else:
                g["days_since_salary"] = (g["as_of_date"] - pd.Timestamp(lsd)).dt.days.clip(0, DEFAULT_DAYS_SINCE_SALARY_MISSING).astype(float)

            frames.append(g)

    feats = pd.concat(frames, ignore_index=True)

    if has_entity:
        keep = [
            "entity_code", "customer_id", "as_of_date", "customer_type", "segment",
            "avg_outflow_30", "p95_outflow_30", "outflow_days_30", "netflow_mean_30",
            "outflow_std_30", "max_single_outflow_30",
            "avg_bal_30", "std_bal_30", "min_bal_30",
            "days_since_salary",
        ]
    else:
        keep = [
            "customer_id", "as_of_date", "customer_type", "segment",
            "avg_outflow_30", "p95_outflow_30", "outflow_days_30", "netflow_mean_30",
            "outflow_std_30", "max_single_outflow_30",
            "avg_bal_30", "std_bal_30", "min_bal_30",
            "days_since_salary",
        ]
    return feats[keep].rename(columns={"as_of_date": "obs_date"})
