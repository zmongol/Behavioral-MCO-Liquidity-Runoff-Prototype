#!/usr/bin/env python3
# make_mco_report.py

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

from mco.config import LOG_LEVEL, TOP_CONTRIBUTORS_N, TAIL_CAPTURE_PCT

logger = logging.getLogger(__name__)


def mco_horizon_actual_summary(labels_long: pd.DataFrame) -> pd.DataFrame:
    df = labels_long.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    if "entity_code" in df.columns:
        return (
            df.groupby(["entity_code", "obs_date", "horizon_bucket", "horizon_days"], as_index=False)
              .agg(
                  n_customers=("customer_id", "nunique"),
                  total_actual=("mco_amount_sgd", "sum"),
              )
              .sort_values(["entity_code", "obs_date", "horizon_days"])
        )
    else:
        return (
            df.groupby(["obs_date", "horizon_bucket", "horizon_days"], as_index=False)
              .agg(
                  n_customers=("customer_id", "nunique"),
                  total_actual=("mco_amount_sgd", "sum"),
              )
              .sort_values(["obs_date", "horizon_days"])
        )


def mco_horizon_pred_summary(pred_long: pd.DataFrame) -> pd.DataFrame:
    df = pred_long.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    if "entity_code" in df.columns:
        return (
            df.groupby(["entity_code", "obs_date", "horizon_bucket", "horizon_days"], as_index=False)
              .agg(
                  n_customers=("customer_id", "nunique"),
                  total_actual=("mco_actual_sgd", "sum"),
                  total_p50=("mco_pred_p50_sgd", "sum"),
                  total_p90=("mco_pred_p90_sgd", "sum"),
              )
              .sort_values(["entity_code", "obs_date", "horizon_days"])
        )
    else:
        return (
            df.groupby(["obs_date", "horizon_bucket", "horizon_days"], as_index=False)
              .agg(
                  n_customers=("customer_id", "nunique"),
                  total_actual=("mco_actual_sgd", "sum"),
                  total_p50=("mco_pred_p50_sgd", "sum"),
                  total_p90=("mco_pred_p90_sgd", "sum"),
              )
              .sort_values(["obs_date", "horizon_days"])
        )


def tail_capture_by_horizon(pred_long: pd.DataFrame, tail_pct: float = None) -> pd.DataFrame:
    if tail_pct is None:
        tail_pct = TAIL_CAPTURE_PCT
    df = pred_long.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    rows = []
    if "entity_code" in df.columns:
        for (entity, dt, bucket, hdays), g in df.groupby(["entity_code", "obs_date", "horizon_bucket", "horizon_days"], sort=True):
            n = g["customer_id"].nunique()
            if n == 0:
                continue
            k = max(1, int(round(tail_pct * n)))

            top_actual = set(g.sort_values("mco_actual_sgd", ascending=False).head(k)["customer_id"])
            top_pred = set(g.sort_values("risk_score", ascending=False).head(k)["customer_id"])
            hits = len(top_actual & top_pred)

            rows.append({
                "entity_code": entity,
                "obs_date": dt.date().isoformat(),
                "horizon_bucket": bucket,
                "horizon_days": int(hdays),
                "n_customers": int(n),
                "k": int(k),
                "tail_pct": float(tail_pct),
                "hits": int(hits),
                "tail_capture_rate": float(hits / k),
            })
    else:
        for (dt, bucket, hdays), g in df.groupby(["obs_date", "horizon_bucket", "horizon_days"], sort=True):
            n = g["customer_id"].nunique()
            if n == 0:
                continue
            k = max(1, int(round(tail_pct * n)))

            top_actual = set(g.sort_values("mco_actual_sgd", ascending=False).head(k)["customer_id"])
            top_pred = set(g.sort_values("risk_score", ascending=False).head(k)["customer_id"])
            hits = len(top_actual & top_pred)

            rows.append({
                "obs_date": dt.date().isoformat(),
                "horizon_bucket": bucket,
                "horizon_days": int(hdays),
                "n_customers": int(n),
                "k": int(k),
                "tail_pct": float(tail_pct),
                "hits": int(hits),
                "tail_capture_rate": float(hits / k),
            })

    return pd.DataFrame(rows)


def top_contributors_by_horizon_latest(pred_long: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    if top_n is None:
        top_n = TOP_CONTRIBUTORS_N
    df = pred_long.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    
    out = []
    if "entity_code" in df.columns:
        for entity in df["entity_code"].unique():
            entity_df = df[df["entity_code"] == entity]
            latest = entity_df["obs_date"].max()
            g = entity_df[entity_df["obs_date"] == latest].copy()

            for (bucket, days), gg in g.groupby(["horizon_bucket", "horizon_days"], sort=True):
                top = gg.sort_values("risk_score", ascending=False).head(top_n).copy()
                top.insert(0, "as_of_latest", latest.date().isoformat())
                out.append(top)
    else:
        latest = df["obs_date"].max()
        g = df[df["obs_date"] == latest].copy()

        for (bucket, days), gg in g.groupby(["horizon_bucket", "horizon_days"], sort=True):
            top = gg.sort_values("risk_score", ascending=False).head(top_n).copy()
            top.insert(0, "as_of_latest", latest.date().isoformat())
            out.append(top)
    
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to mco_labels_long.csv (all horizons actual)")
    ap.add_argument("--pred", required=True, help="Path to predictions_long.csv (trained horizons predictions)")
    ap.add_argument("--out", required=True, help="Output folder for reports")
    args = ap.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    try:
        logger.info("Starting MCO report generation")
        labels_path = Path(args.labels).expanduser().resolve()
        pred_path = Path(args.pred).expanduser().resolve()
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")

        logger.info(f"Loading labels from {labels_path}")
        labels_long = pd.read_csv(labels_path)
        logger.info(f"Loading predictions from {pred_path}")
        pred_long = pd.read_csv(pred_path)

        # Get list of entities
        has_entities = "entity_code" in labels_long.columns
        if has_entities:
            entities = sorted(labels_long["entity_code"].unique())
            logger.info(f"Found {len(entities)} entities: {entities}")
        else:
            entities = []
            logger.info("Single-entity dataset (no entity_code)")

        if has_entities:
            # Generate overall reports (all entities combined)
            logger.info("Computing overall (cross-entity) actual summary...")
            actual_s = mco_horizon_actual_summary(labels_long)
            actual_s.to_csv(out_dir / "mco_horizon_actual_summary_overall.csv", index=False)

            logger.info("Computing overall (cross-entity) prediction summary...")
            pred_s = mco_horizon_pred_summary(pred_long)
            pred_s.to_csv(out_dir / "mco_horizon_pred_summary_overall.csv", index=False)

            logger.info(f"Computing overall tail capture ({TAIL_CAPTURE_PCT*100:.1f}%)...")
            tail = tail_capture_by_horizon(pred_long)
            tail.to_csv(out_dir / "tail_capture_by_horizon_overall.csv", index=False)

            logger.info(f"Computing overall top {TOP_CONTRIBUTORS_N} contributors...")
            top = top_contributors_by_horizon_latest(pred_long)
            top.to_csv(out_dir / "top_contributors_by_horizon_overall.csv", index=False)

            # Generate per-entity reports
            for entity in entities:
                entity_dir = out_dir / entity
                entity_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Generating reports for entity: {entity}")

                # Filter to this entity
                labels_entity = labels_long[labels_long["entity_code"] == entity]
                pred_entity = pred_long[pred_long["entity_code"] == entity]

                # 1) Actual summary for entity
                actual_s_entity = mco_horizon_actual_summary(labels_entity)
                actual_s_entity.to_csv(entity_dir / "mco_horizon_actual_summary.csv", index=False)

                # 2) Pred summary for entity
                pred_s_entity = mco_horizon_pred_summary(pred_entity)
                pred_s_entity.to_csv(entity_dir / "mco_horizon_pred_summary.csv", index=False)

                # 3) Tail capture for entity
                tail_entity = tail_capture_by_horizon(pred_entity)
                tail_entity.to_csv(entity_dir / "tail_capture_by_horizon.csv", index=False)

                # 4) Top contributors for entity
                top_entity = top_contributors_by_horizon_latest(pred_entity)
                top_entity.to_csv(entity_dir / "top_contributors_by_horizon.csv", index=False)

                # 5) Entity-level metrics
                overall_entity = {
                    "entity_code": entity,
                    "labels_rows": int(len(labels_entity)),
                    "pred_rows": int(len(pred_entity)),
                    "n_customers_labels": int(labels_entity["customer_id"].nunique()),
                    "n_customers_pred": int(pred_entity["customer_id"].nunique()),
                    "latest_obs_date_pred": str(pd.to_datetime(pred_entity["obs_date"]).max().date()) if len(pred_entity) > 0 else None,
                    "tail_capture_avg_trained_horizons": float(tail_entity["tail_capture_rate"].mean()) if len(tail_entity) > 0 else None,
                }
                (entity_dir / "entity_metrics.json").write_text(json.dumps(overall_entity, indent=2))

                # 6) Entity text report
                latest = pd.to_datetime(pred_entity["obs_date"]).max() if len(pred_entity) > 0 else None
                latest_tbl = pred_s_entity[pred_s_entity["obs_date"] == latest].copy() if latest else pd.DataFrame()

                lines = []
                lines.append(f"MCO MULTI-HORIZON REPORT - ENTITY: {entity}\n")
                lines.append(f"Labels: {labels_path}\n")
                lines.append(f"Predictions: {pred_path}\n")
                lines.append(f"Latest obs_date (pred): {latest.date().isoformat() if latest else 'N/A'}\n")
                if overall_entity["tail_capture_avg_trained_horizons"] is not None:
                    lines.append(f"Tail capture avg (trained horizons): {overall_entity['tail_capture_avg_trained_horizons']:.4f}\n")
                lines.append(f"Customers (labels): {overall_entity['n_customers_labels']}\n")
                lines.append(f"Customers (pred): {overall_entity['n_customers_pred']}\n")

                if len(latest_tbl) > 0:
                    lines.append("\nTRAINED HORIZON TOTALS (latest date)\n")
                    lines.append(latest_tbl[["horizon_bucket", "horizon_days", "total_p50", "total_p90", "total_actual"]].to_string(index=False))

                (entity_dir / "report.txt").write_text("\n".join(lines))
        else:
            # Single-entity dataset: generate reports at root level
            logger.info("Computing actual summary...")
            actual_s = mco_horizon_actual_summary(labels_long)
            actual_s.to_csv(out_dir / "mco_horizon_actual_summary.csv", index=False)

            logger.info("Computing prediction summary...")
            pred_s = mco_horizon_pred_summary(pred_long)
            pred_s.to_csv(out_dir / "mco_horizon_pred_summary.csv", index=False)

            logger.info(f"Computing tail capture ({TAIL_CAPTURE_PCT*100:.1f}%)...")
            tail = tail_capture_by_horizon(pred_long)
            tail.to_csv(out_dir / "tail_capture_by_horizon.csv", index=False)

            logger.info(f"Computing top {TOP_CONTRIBUTORS_N} contributors...")
            top = top_contributors_by_horizon_latest(pred_long)
            top.to_csv(out_dir / "top_contributors_by_horizon.csv", index=False)

            # Metrics
            latest = pd.to_datetime(pred_long["obs_date"]).max() if len(pred_long) > 0 else None
            latest_tbl = pred_s[pred_s["obs_date"] == latest].copy() if latest else pd.DataFrame()

            overall = {
                "labels_rows": int(len(labels_long)),
                "pred_rows": int(len(pred_long)),
                "n_customers_labels": int(labels_long["customer_id"].nunique()),
                "n_customers_pred": int(pred_long["customer_id"].nunique()),
                "latest_obs_date_pred": str(latest.date().isoformat()) if latest else None,
                "tail_capture_avg_trained_horizons": float(tail["tail_capture_rate"].mean()) if len(tail) else None,
            }
            (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2))

            # Text report
            lines = []
            lines.append("MCO MULTI-HORIZON REPORT\n")
            lines.append(f"Labels: {labels_path}\n")
            lines.append(f"Predictions: {pred_path}\n")
            lines.append(f"Latest obs_date (pred): {latest.date().isoformat() if latest else 'N/A'}\n")
            if overall["tail_capture_avg_trained_horizons"] is not None:
                lines.append(f"Tail capture avg (trained horizons): {overall['tail_capture_avg_trained_horizons']:.4f}\n")

            if len(latest_tbl) > 0:
                lines.append("\nTRAINED HORIZON TOTALS (latest date)\n")
                lines.append(latest_tbl[["horizon_bucket", "horizon_days", "total_p50", "total_p90", "total_actual"]].to_string(index=False))

            (out_dir / "report.txt").write_text("\n".join(lines))

        logger.info("✅ Report generation completed")
        if has_entities:
            print(f"✅ Wrote reports to: {out_dir}")
            print(f"   - Overall: {out_dir}/*_overall.csv")
            print(f"   - Per-entity: {out_dir}/{{{','.join(entities)}}}/")
        else:
            print(f"✅ Wrote reports to: {out_dir}")
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
