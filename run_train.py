#!/usr/bin/env python3
# run_train.py

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mco.config import LOG_LEVEL
from mco.io_extract import unzip_to_dir, load_tables
from mco.transform_daily import build_customer_daily
from mco.labels import compute_mco_labels_multi
from mco.features import build_features
from mco.model import train_two_stage_multi
from mco.data_validation import validate_customer_daily, validate_labels, validate_features, profile_data

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, required=True, help="Path to dataset zip")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--mode", type=str, default="report", choices=["report", "train"], 
                    help="Label mode: 'report' (all horizons) or 'train' (only training horizons)")
    args = ap.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    zip_path = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting MCO training pipeline")
        logger.info(f"Input ZIP: {zip_path}")
        logger.info(f"Output directory: {out_dir}")

        data_dir = out_dir / "extracted"
        unzip_to_dir(zip_path, data_dir)
        cust, fx, txn, bal = load_tables(data_dir)

        # Core fact: customer_day
        logger.info("Building customer-daily dataset...")
        customer_daily = build_customer_daily(cust, fx, txn, bal)
        logger.info(f"Built customer-daily with {len(customer_daily)} rows across {customer_daily['customer_id'].nunique()} customers")

        # Validate customer_daily
        val_err = validate_customer_daily(customer_daily)
        if val_err:
            raise ValueError(f"customer_daily validation failed: {val_err}")

        # Labels: compute based on mode
        logger.info(f"Computing MCO labels ({args.mode} mode)...")
        labels_long = compute_mco_labels_multi(customer_daily, mode=args.mode)
        labels_long.to_csv(out_dir / "mco_labels_long.csv", index=False)
        logger.info(f"Computed {len(labels_long)} label rows")

        # Validate labels
        val_err = validate_labels(labels_long)
        if val_err:
            raise ValueError(f"labels validation failed: {val_err}")

        # Features: shared across horizons
        logger.info("Building rolling features...")
        feats = build_features(customer_daily, txn)
        logger.info(f"Built {len(feats)} feature rows with {len(feats.columns)} columns")

        # Validate features
        val_err = validate_features(feats)
        if val_err:
            raise ValueError(f"features validation failed: {val_err}")

        # Profile and log data summary
        profile = profile_data(customer_daily, labels_long, feats)
        logger.info(f"Data profile: {profile}")

        # Train table: labels + features
        logger.info("Merging labels and features...")
        if "entity_code" in labels_long.columns:
            train_long = labels_long.merge(feats, on=["entity_code", "customer_id", "obs_date"], how="left")
        else:
            train_long = labels_long.merge(feats, on=["customer_id", "obs_date"], how="left")
        missing_feats = train_long.isna().sum().sum()
        logger.warning(f"Merged table has {missing_feats} missing feature values (will be dropped during training)")

        # Train models
        logger.info("Training two-stage models...")
        metrics = train_two_stage_multi(train_long, out_dir)

        logger.info("✅ Pipeline completed successfully")
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
