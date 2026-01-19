# mco/transform_daily.py
"""
Transform raw BALANCE_DAILY + TXN_LEDGER into a clean customer-day spine.

Output: one row per (customer_id, as_of_date) with:
- base_balance_eod (SGD)
- inflow_sgd, outflow_sgd, netflow_sgd (net outflow)
- customer_type, segment
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .config import BASE_CCY

logger = logging.getLogger(__name__)


def to_base_ccy(df: pd.DataFrame, fx: pd.DataFrame, date_col: str, ccy_col: str, amt_col: str) -> pd.Series:
    """
    Convert amounts to BASE_CCY using FX_RATES_DAILY.
    If missing, assume fx_rate = 1.0.
    """
    fx_base = fx[fx["to_currency"] == BASE_CCY].rename(columns={"date": date_col, "from_currency": ccy_col})
    tmp = df[[date_col, ccy_col, amt_col]].merge(
        fx_base[[date_col, ccy_col, "fx_rate"]],
        on=[date_col, ccy_col],
        how="left",
    )
    rate = tmp["fx_rate"].fillna(1.0).astype(float)
    return tmp[amt_col].astype(float) * rate


def build_customer_daily(cust: pd.DataFrame, fx: pd.DataFrame, txn: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (entity_code, customer_id, as_of_date) or just (customer_id, as_of_date) if no entities:
      - base_balance_eod (SGD)
      - inflow_sgd / outflow_sgd / netflow_sgd (net OUTflow)
      - customer_type, segment, entity_code (if present), entity_name (if present)
    """
    # Check if entity_code exists
    has_entity = "entity_code" in bal.columns
    
    # ---- Balances to SGD ----
    bal = bal.copy()
    bal["bal_sgd"] = to_base_ccy(bal, fx, "as_of_date", "currency", "end_of_day_balance")

    if has_entity:
        cust_bal = (
            bal.groupby(["as_of_date", "customer_id", "entity_code"], as_index=False)["bal_sgd"].sum()
            .rename(columns={"bal_sgd": "base_balance_eod"})
        )
    else:
        cust_bal = (
            bal.groupby(["as_of_date", "customer_id"], as_index=False)["bal_sgd"].sum()
            .rename(columns={"bal_sgd": "base_balance_eod"})
        )

    # ---- Transactions to SGD ----
    txn = txn.copy()
    txn["txn_date"] = txn["txn_datetime"].dt.normalize()
    txn["amt_sgd"] = to_base_ccy(txn, fx, "txn_date", "currency", "amount")

    # Signed flow: CR +, DR -
    drcr = txn["dr_cr_flag"].astype(str).str.upper()
    signed = np.where(drcr == "CR", txn["amt_sgd"].values, -txn["amt_sgd"].values)

    # Exclude SELF transfers
    cpty = txn["counterparty_type"].astype(str).str.upper()
    mask = cpty != "SELF"
    txn = txn.loc[mask].copy()
    txn["signed_sgd"] = signed[mask]

    txn["inflow_sgd"] = np.where(txn["signed_sgd"] > 0, txn["signed_sgd"], 0.0)
    txn["outflow_sgd"] = np.where(txn["signed_sgd"] < 0, -txn["signed_sgd"], 0.0)

    if has_entity:
        flows = (
            txn.groupby(["txn_date", "customer_id", "entity_code"], as_index=False)[["inflow_sgd", "outflow_sgd"]].sum()
            .rename(columns={"txn_date": "as_of_date"})
        )
    else:
        flows = (
            txn.groupby(["txn_date", "customer_id"], as_index=False)[["inflow_sgd", "outflow_sgd"]].sum()
            .rename(columns={"txn_date": "as_of_date"})
        )
    # netflow is net outflow
    flows["netflow_sgd"] = flows["outflow_sgd"] - flows["inflow_sgd"]

    # ---- Merge balances as spine ----
    if has_entity:
        df = cust_bal.merge(flows, on=["as_of_date", "customer_id", "entity_code"], how="left")
    else:
        df = cust_bal.merge(flows, on=["as_of_date", "customer_id"], how="left")
    df[["inflow_sgd", "outflow_sgd", "netflow_sgd"]] = df[["inflow_sgd", "outflow_sgd", "netflow_sgd"]].fillna(0.0)

    # Attach segment/type and entity info
    if has_entity:
        df = df.merge(cust[["customer_id", "customer_type", "segment", "entity_code", "entity_name"]], 
                      on=["customer_id", "entity_code"], how="left")
        logger.info(f"Built customer-daily dataset: {len(df)} rows, {df['customer_id'].nunique()} customers, "
                    f"{df['entity_code'].nunique()} entities, {df['as_of_date'].min().date()} to {df['as_of_date'].max().date()}")
        return df.sort_values(["entity_code", "customer_id", "as_of_date"]).reset_index(drop=True)
    else:
        df = df.merge(cust[["customer_id", "customer_type", "segment"]], on="customer_id", how="left")
        logger.info(f"Built customer-daily dataset: {len(df)} rows, {df['customer_id'].nunique()} customers, "
                    f"{df['as_of_date'].min().date()} to {df['as_of_date'].max().date()}")
        return df.sort_values(["customer_id", "as_of_date"]).reset_index(drop=True)
