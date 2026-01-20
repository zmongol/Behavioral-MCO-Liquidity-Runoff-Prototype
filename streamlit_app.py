#!/usr/bin/env python3
"""Simple Streamlit dashboard for exploring MCO runs."""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_SCRIPT = PROJECT_ROOT / "run_all.sh"

st.set_page_config(page_title="MCO Demo Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def list_run_dirs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    return sorted(run_dirs, key=lambda p: p.name, reverse=True)


@st.cache_data(show_spinner=False)
def load_csv(path_str: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path_str, parse_dates=parse_dates or [])


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data(show_spinner=False)
def load_predictions(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str, parse_dates=["obs_date"]) if Path(path_str).exists() else pd.DataFrame()


def trigger_demo_run(mode: str = "small") -> Tuple[int, str]:
    cmd = [str(RUN_SCRIPT), mode]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return res.returncode, output


def available_entities(report_dir: Path) -> List[str]:
    if not report_dir.exists():
        return []
    return sorted([p.name for p in report_dir.iterdir() if p.is_dir()])


def resolve_report_file(report_dir: Path, entity: Optional[str], overall_name: str, single_name: str) -> Optional[Path]:
    if entity and entity != "All entities":
        candidate = report_dir / entity / single_name
        return candidate if candidate.exists() else None
    for candidate in [report_dir / overall_name, report_dir / single_name]:
        if candidate.exists():
            return candidate
    return None


def aggregate_entity(df: pd.DataFrame, entity: Optional[str], agg: str = "sum") -> pd.DataFrame:
    if df.empty:
        return df
    has_entity = "entity_code" in df.columns
    if entity and entity != "All entities" and has_entity:
        return df[df["entity_code"] == entity].copy()
    if has_entity:
        group_cols = [c for c in ["obs_date", "horizon_bucket", "horizon_days"] if c in df.columns]
        value_cols = [c for c in df.select_dtypes(include="number").columns if c not in group_cols]
        if not value_cols:
            return df.drop(columns=["entity_code"]).drop_duplicates(subset=group_cols)
        agg_fn = "mean" if agg == "mean" else "sum"
        grouped = (
            df.drop(columns=["entity_code"])
              .groupby(group_cols, as_index=False)
              .agg({c: agg_fn for c in value_cols})
        )
        return grouped
    return df


runs = list_run_dirs()

st.title("Behavioral MCO Demo Dashboard")
st.caption("Load a completed run from the pipeline and explore tail capture, totals, and top risk accounts.")

with st.sidebar:
    st.header("Run selection")
    run_label = st.selectbox("Available runs", options=runs, format_func=lambda p: p.name if isinstance(p, Path) else p, index=0 if runs else None, placeholder="No runs yet")

    st.divider()
    st.subheader("Create a demo run")
    new_mode = st.selectbox("Mode", options=["small", "big"], index=0)
    if st.button("Run pipeline now", use_container_width=True):
        with st.spinner("Running pipeline... this may take a minute"):
            code, logs = trigger_demo_run(mode=new_mode)
        if code == 0:
            st.success("Pipeline completed. Refresh the run picker above.")
            list_run_dirs.clear()
            st.text_area("Logs", logs, height=240)
        else:
            st.error("Pipeline failed — see logs below.")
            st.text_area("Logs", logs, height=240)

if not run_label:
    st.info("No runs found. Use the sidebar to create a demo run (small is quickest) or run ./run_all.sh manually, then refresh.")
    st.stop()

run_dir = Path(run_label)
outputs_dir = run_dir / "outputs"
reports_dir = run_dir / "reports"
meta_dir = run_dir / "metadata"

run_meta = load_json(str(meta_dir / "run_metadata.json")) if (meta_dir / "run_metadata.json").exists() else {}
metrics_json = load_json(str(outputs_dir / "metrics_by_horizon.json")) if (outputs_dir / "metrics_by_horizon.json").exists() else {}
metrics_df = pd.DataFrame(metrics_json).T.reset_index().rename(columns={"index": "horizon_bucket"}) if metrics_json else pd.DataFrame()
if not metrics_df.empty and "horizon_days" in metrics_df.columns:
    metrics_df["horizon_days"] = metrics_df["horizon_days"].astype(int)

entities = available_entities(reports_dir)
entity_choice = None
if entities:
    entity_choice = st.selectbox("Entity", options=["All entities"] + entities, index=0)

# Resolve report files based on entity selection
pred_summary_path = resolve_report_file(reports_dir, entity_choice, "mco_horizon_pred_summary_overall.csv", "mco_horizon_pred_summary.csv")
tail_capture_path = resolve_report_file(reports_dir, entity_choice, "tail_capture_by_horizon_overall.csv", "tail_capture_by_horizon.csv")

pred_summary_df = load_csv(str(pred_summary_path), parse_dates=["obs_date"]) if pred_summary_path else pd.DataFrame()
pred_summary_df = aggregate_entity(pred_summary_df, entity_choice, agg="sum")
tail_df_raw = load_csv(str(tail_capture_path), parse_dates=["obs_date"]) if tail_capture_path else pd.DataFrame()
tail_df = aggregate_entity(tail_df_raw, entity_choice, agg="mean")
if not pred_summary_df.empty and "horizon_days" in pred_summary_df.columns:
    pred_summary_df["horizon_days"] = pred_summary_df["horizon_days"].astype(int)
if not tail_df.empty and "horizon_days" in tail_df.columns:
    tail_df["horizon_days"] = tail_df["horizon_days"].astype(int)

pred_df = load_predictions(str(outputs_dir / "predictions_long.csv"))
if entity_choice and "entity_code" in pred_df.columns:
    pred_df = pred_df[pred_df["entity_code"] == entity_choice] if entity_choice != "All entities" else pred_df
if not pred_df.empty and "horizon_days" in pred_df.columns:
    pred_df["horizon_days"] = pred_df["horizon_days"].astype(int)

st.subheader(f"Run: {run_dir.name}")
col1, col2, col3 = st.columns(3)
col1.metric("Mode", run_meta.get("mode", "?"))
col2.metric("ZIP", Path(run_meta.get("zip", "?")).name if run_meta else "?")
col3.metric("Python", run_meta.get("python", "?"))

if metrics_df.empty:
    st.warning("metrics_by_horizon.json not found — run the pipeline to generate predictions.")
else:
    st.write("Metrics by horizon")
    sort_col = "horizon_days" if "horizon_days" in metrics_df.columns else None
    st.dataframe(metrics_df.sort_values(sort_col) if sort_col else metrics_df, use_container_width=True)

if tail_df.empty or pred_summary_df.empty:
    st.warning("Report CSVs not found yet. Ensure the pipeline completed successfully.")
    st.stop()

latest_tail_date = tail_df["obs_date"].max()
latest_tail = tail_df[tail_df["obs_date"] == latest_tail_date].sort_values("horizon_days")
latest_pred_date = pred_summary_df["obs_date"].max()
pred_latest = pred_summary_df[pred_summary_df["obs_date"] == latest_pred_date].copy()

st.markdown("---")
st.subheader("Tail capture and totals")
col_a, col_b = st.columns(2)
with col_a:
    st.caption(f"Tail capture (latest: {latest_tail_date.date()})")
    chart_df = latest_tail.set_index("horizon_days")["tail_capture_rate"]
    st.bar_chart(chart_df)
    st.dataframe(latest_tail, use_container_width=True)
with col_b:
    st.caption(f"Predicted vs actual totals (latest: {latest_pred_date.date()})")
    cols_to_show = [c for c in ["horizon_bucket", "horizon_days", "total_actual", "total_p50", "total_p90"] if c in pred_latest.columns]
    st.dataframe(pred_latest[cols_to_show].sort_values("horizon_days"), use_container_width=True)

st.markdown("---")
st.subheader("Top risk accounts")
if pred_df.empty:
    st.info("predictions_long.csv not found; cannot show top accounts.")
else:
    if "horizon_days" in pred_df.columns:
        horizon_order = pred_df[["horizon_bucket", "horizon_days"]].drop_duplicates().sort_values("horizon_days")
        horizons = horizon_order["horizon_bucket"].tolist()
    else:
        horizons = sorted(pred_df["horizon_bucket"].unique())
    selected_h = st.selectbox("Horizon", options=horizons)
    subset_h = pred_df[pred_df["horizon_bucket"] == selected_h].copy()
    subset_h["obs_date"] = pd.to_datetime(subset_h["obs_date"])
    date_options = [pd.to_datetime(d).to_pydatetime() for d in sorted(subset_h["obs_date"].unique())]
    picked_date = st.selectbox(
        "Observation date",
        options=date_options,
        index=len(date_options) - 1 if date_options else 0,
        format_func=lambda d: d.date().isoformat(),
    ) if date_options else None
    top_n = st.slider("Show top N by risk_score", min_value=5, max_value=100, value=30, step=5)

    if picked_date is not None:
        view = subset_h[subset_h["obs_date"] == pd.to_datetime(picked_date)].copy()
        view = view.sort_values("risk_score", ascending=False).head(top_n)
        cols = [c for c in ["entity_code", "customer_id", "horizon_bucket", "horizon_days", "obs_date", "risk_score", "p_runoff", "mco_pred_p90_sgd", "mco_pred_p50_sgd", "mco_actual_sgd", "base_balance_eod"] if c in view.columns]
        st.dataframe(view[cols], use_container_width=True)
    else:
        st.info("No prediction rows found for the selected filters.")

st.markdown("---")
st.caption("Run the pipeline again after changing code to refresh these visuals.")
