# mco/labels.py
"""
Compute multi-horizon MCO labels in LONG format.

Definition:
MCO_H(t0) = max_{k=1..H} sum_{d=1..k} netflow(t0 + d)

Output (long):
customer_id, obs_date, horizon_bucket, horizon_days, mco_amount_sgd
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .config import HORIZON_BUCKETS_REPORT, HORIZON_BUCKETS_TRAIN

logger = logging.getLogger(__name__)


def _mco_for_horizon_series(netflow: np.ndarray, H: int) -> np.ndarray:
    """
    netflow aligned to obs_date index.
    For each i, use future window netflow[i+1 : i+H+1].
    Returns array of length n with NaN for last H rows (insufficient future).
    """
    n = netflow.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= H:
        return out

    # future starts at next day
    base = netflow[1:]  # length n-1
    rows = n - H  # number of obs_date rows with full horizon

    # window matrix shape: (rows, H)
    mat = np.lib.stride_tricks.sliding_window_view(base, window_shape=H)[:rows, :]
    cum = np.cumsum(mat, axis=1)
    out[:rows] = np.maximum(0.0, np.max(cum, axis=1))  # Fix A: clamp MCO to non-negative
    return out


def compute_mco_labels_multi(customer_daily: pd.DataFrame, mode: str = "report") -> pd.DataFrame:
    """
    customer_daily required cols:
      customer_id, as_of_date, netflow_sgd, base_balance_eod
      (optional: entity_code for multi-entity grouping)

    mode: "report" (all horizons for reporting) or "train" (only training horizons for big runs)
    returns long labels for specified horizons, grouped by entity if present.
    """
    # Fix B: use mode-dependent horizon buckets
    horizon_buckets = HORIZON_BUCKETS_TRAIN if mode == "train" else HORIZON_BUCKETS_REPORT
    has_entity = "entity_code" in customer_daily.columns
    
    frames = []
    if has_entity:
        for entity, entity_group in customer_daily.groupby("entity_code", sort=False):
            for cid, g in entity_group.groupby("customer_id", sort=False):
                g = g.sort_values("as_of_date").reset_index(drop=True)
                nf = g["netflow_sgd"].to_numpy(dtype=np.float64)

                for bucket, H in horizon_buckets.items():
                    mco = _mco_for_horizon_series(nf, int(H))
                    dfh = pd.DataFrame({
                        "entity_code": entity,
                        "customer_id": cid,
                        "obs_date": g["as_of_date"].values,
                        "horizon_bucket": bucket,
                        "horizon_days": int(H),
                        "mco_amount_sgd": mco,
                        "base_balance_eod": g["base_balance_eod"].values,
                    })
                    frames.append(dfh)
    else:
        for cid, g in customer_daily.groupby("customer_id", sort=False):
            g = g.sort_values("as_of_date").reset_index(drop=True)
            nf = g["netflow_sgd"].to_numpy(dtype=np.float64)

            for bucket, H in horizon_buckets.items():
                mco = _mco_for_horizon_series(nf, int(H))
                dfh = pd.DataFrame({
                    "customer_id": cid,
                    "obs_date": g["as_of_date"].values,
                    "horizon_bucket": bucket,
                    "horizon_days": int(H),
                    "mco_amount_sgd": mco,
                    "base_balance_eod": g["base_balance_eod"].values,
                })
                frames.append(dfh)

    labels = pd.concat(frames, ignore_index=True)
    before_drop = len(labels)
    labels = labels.dropna(subset=["mco_amount_sgd"]).reset_index(drop=True)
    logger.info(f"Computed MCO labels: {before_drop} rows total, {len(labels)} valid (dropped {before_drop - len(labels)} NaN rows)")
    return labels
