# mco/model.py
"""
Train two-stage models for TRAIN horizons only (5),
but allow reporting for all horizons using labels.

Writes:
- predictions_long.csv (trained horizons only)
- metrics_by_horizon.json
- models/model_<bucket>_*.joblib
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
import xgboost as xgb

from .config import (
    HORIZON_BUCKETS_TRAIN,
    P90_PROB_FLOOR,
    RISK_SCORE_POWER,
    RUNOFF_ABS_THRESHOLD,
    RUNOFF_REL_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    LOGREG_SOLVER,
    LOGREG_MAX_ITER,
    RIDGE_ALPHA,
    QREG_QUANTILE,
    QREG_ALPHA,
    QREG_SOLVER,
    MIN_RUNOFF_CASES_FOR_STAGE2,
)

logger = logging.getLogger(__name__)


def _preprocess_features(df, num_cols, cat_cols):
    """Preprocess features: scale numeric, one-hot encode categorical."""
    X = df[num_cols + cat_cols].copy()
    
    # Scale numeric features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # One-hot encode categorical
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = encoder.fit_transform(X[cat_cols])
    X_cat_names = encoder.get_feature_names_out(cat_cols)
    
    X_processed = np.column_stack([X[num_cols].values, cat_encoded])
    feature_names = list(num_cols) + list(X_cat_names)
    
    return X_processed, feature_names, scaler, encoder

def _transform_features(df, num_cols, cat_cols, scaler, encoder):
    """Apply saved preprocessors to new data."""
    X = df[num_cols + cat_cols].copy()
    X[num_cols] = scaler.transform(X[num_cols])
    cat_encoded = encoder.transform(X[cat_cols])
    X_processed = np.column_stack([X[num_cols].values, cat_encoded])
    return X_processed


def train_two_stage_multi(train_long: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect GPU availability
    try:
        gpu_device = "cuda"
        xgb.DMatrix([[0]], label=[0])  # Quick GPU test
    except:
        gpu_device = "cpu"
        logger.info("GPU not available, using CPU")

    df = train_long.copy()

    # Keep only TRAIN horizons for model training
    df = df[df["horizon_bucket"].isin(HORIZON_BUCKETS_TRAIN.keys())].copy()

    # Require rolling history features
    df = df.dropna(subset=["avg_outflow_30", "avg_bal_30"]).copy()
    logger.info(f"After feature filtering: {len(df)} rows (from {len(train_long)})")

    if len(df) == 0:
        logger.error("No rows remain after feature filtering. Cannot train models.")
        raise ValueError("Training data is empty after feature filtering")

    num_cols = [
        "avg_outflow_30", "p95_outflow_30", "outflow_days_30", "netflow_mean_30",
        "outflow_std_30", "max_single_outflow_30",
        "avg_bal_30", "std_bal_30", "min_bal_30", "days_since_salary",
    ]
    cat_cols = ["customer_type", "segment"]

    # Split by obs_date (time-proxy split) – shared across horizons
    gss = GroupShuffleSplit(n_splits=1, test_size=float(TEST_SIZE), random_state=int(RANDOM_STATE))
    groups = df["obs_date"].astype(str).values
    X_all = df[num_cols + cat_cols]
    y_dummy = np.zeros(len(df), dtype=int)
    tr_idx, te_idx = next(gss.split(X_all, y_dummy, groups=groups))

    df_tr = df.iloc[tr_idx].copy()
    df_te = df.iloc[te_idx].copy()

    all_pred_rows = []
    metrics = {}

    for bucket, H in HORIZON_BUCKETS_TRAIN.items():
        g_tr = df_tr[df_tr["horizon_bucket"] == bucket].copy()
        g_te = df_te[df_te["horizon_bucket"] == bucket].copy()
        if len(g_tr) == 0 or len(g_te) == 0:
            logger.warning(f"Skipping horizon {bucket}: train={len(g_tr)} rows, test={len(g_te)} rows")
            continue

        # Stage-1 classification target for this horizon
        thr_tr = np.maximum(
            float(RUNOFF_ABS_THRESHOLD),
            float(RUNOFF_REL_THRESHOLD) * g_tr["base_balance_eod"].astype(float).values
        )
        y_cls_tr = (g_tr["mco_amount_sgd"].astype(float).values > thr_tr).astype(int)

        # Guard: if no positive labels, skip model training to avoid XGBoost base_score error
        if y_cls_tr.sum() == 0:
            logger.warning(f"Horizon {bucket}: no positive runoff cases in training — emitting zero predictions and skipping models.")

            # Emit zeroed predictions for test split to keep reporting happy
            y_true = g_te["mco_amount_sgd"].astype(float).values
            p_te = np.zeros(len(g_te), dtype=float)
            sev_p50 = np.zeros(len(g_te), dtype=float)
            sev_p90 = np.zeros(len(g_te), dtype=float)
            risk_score = np.zeros(len(g_te), dtype=float)

            m = {
                "horizon_days": int(H),
                "rows_test": int(len(g_te)),
                "positive_rate_test": 0.0,
                "PR_AUC": None,
                "ROC_AUC": None,
                "MAE_MCO_P50": float(mean_absolute_error(y_true, sev_p50)) if len(g_te) > 0 else None,
                "P90_coverage_all": float(np.mean(y_true <= sev_p90)) if len(g_te) > 0 else None,
                "P90_PROB_FLOOR": float(P90_PROB_FLOOR),
                "RISK_SCORE_POWER": float(RISK_SCORE_POWER),
            }
            metrics[bucket] = m

            cols_to_keep = ["customer_id", "obs_date", "horizon_bucket", "horizon_days", "base_balance_eod"]
            if "entity_code" in g_te.columns:
                cols_to_keep.insert(0, "entity_code")
            outp = g_te[cols_to_keep].copy()
            outp["mco_actual_sgd"] = y_true
            outp["p_runoff"] = p_te
            outp["mco_pred_p50_sgd"] = sev_p50
            outp["mco_pred_p90_sgd"] = sev_p90
            outp["risk_score"] = risk_score
            all_pred_rows.append(outp)
            continue

        X_tr = g_tr[num_cols + cat_cols]
        X_te = g_te[num_cols + cat_cols]

        # Stage 1: p_runoff (XGBoost with GPU)
        X_tr_proc, feature_names, scaler_tr, encoder_tr = _preprocess_features(X_tr, num_cols, cat_cols)
        X_te_proc = _transform_features(X_te, num_cols, cat_cols, scaler_tr, encoder_tr)
        
        stage1 = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            device=gpu_device,
            random_state=int(RANDOM_STATE),
            verbosity=0
        )
        stage1.fit(X_tr_proc, y_cls_tr)
        p_te = stage1.predict_proba(X_te_proc)[:, 1]

        # Stage 2: severity on runoff cases only
        runoff_mask = y_cls_tr == 1
        if runoff_mask.sum() < int(MIN_RUNOFF_CASES_FOR_STAGE2):
            logger.warning(f"Horizon {bucket}: only {runoff_mask.sum()} runoff cases (threshold={MIN_RUNOFF_CASES_FOR_STAGE2}). Skipping stage2 models.")
            sev_p50 = np.zeros(len(g_te), dtype=float)
            sev_p90 = np.zeros(len(g_te), dtype=float)
            stage2_p50 = None
            stage2_p90 = None
        else:
            y_sev_tr = np.log1p(g_tr.loc[runoff_mask, "mco_amount_sgd"].astype(float).values)

            # Stage 2 P50 (XGBoost Regressor)
            X_tr_runoff_proc, feat_names_p50, scaler_p50, encoder_p50 = _preprocess_features(X_tr[runoff_mask], num_cols, cat_cols)
            X_te_proc_p50 = _transform_features(X_te, num_cols, cat_cols, scaler_p50, encoder_p50)
            
            stage2_p50 = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                tree_method="hist",
                device=gpu_device,
                random_state=int(RANDOM_STATE),
                verbosity=0
            )
            stage2_p50.fit(X_tr_runoff_proc, y_sev_tr)
            sev_p50 = np.expm1(stage2_p50.predict(X_te_proc_p50)).clip(min=0)

            # Stage 2 P90 (quantile regressor for the 90th percentile)
            stage2_p90 = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                tree_method="hist",
                device=gpu_device,
                objective="reg:quantileerror",
                quantile_alpha=float(QREG_QUANTILE),
                random_state=int(RANDOM_STATE),
                verbosity=0
            )
            stage2_p90.fit(X_tr_runoff_proc, y_sev_tr)
            sev_p90 = np.expm1(stage2_p90.predict(X_te_proc_p50)).clip(min=0)

        # Combine to dollar outputs
        mco_p50 = p_te * sev_p50
        mco_p90 = np.maximum(mco_p50, sev_p90 * np.clip(p_te, float(P90_PROB_FLOOR), 1.0))
        risk_score = (p_te ** float(RISK_SCORE_POWER)) * sev_p90

        # Test metrics
        y_true = g_te["mco_amount_sgd"].astype(float).values
        thr_te = np.maximum(
            float(RUNOFF_ABS_THRESHOLD),
            float(RUNOFF_REL_THRESHOLD) * g_te["base_balance_eod"].astype(float).values
        )
        y_cls_te = (y_true > thr_te).astype(int)

        m = {
            "horizon_days": int(H),
            "rows_test": int(len(g_te)),
            "positive_rate_test": float(y_cls_te.mean()),
            "PR_AUC": float(average_precision_score(y_cls_te, p_te)) if len(np.unique(y_cls_te)) > 1 else None,
            "ROC_AUC": float(roc_auc_score(y_cls_te, p_te)) if len(np.unique(y_cls_te)) > 1 else None,
            "MAE_MCO_P50": float(mean_absolute_error(y_true, mco_p50)),
            "P90_coverage_all": float(np.mean(y_true <= mco_p90)),
            "P90_PROB_FLOOR": float(P90_PROB_FLOOR),
            "RISK_SCORE_POWER": float(RISK_SCORE_POWER),
        }
        metrics[bucket] = m

        # Save models
        joblib.dump(stage1, models_dir / f"model_{bucket}_stage1.joblib")
        if stage2_p50 is not None:
            joblib.dump(stage2_p50, models_dir / f"model_{bucket}_p50.joblib")
        if stage2_p90 is not None:
            joblib.dump(stage2_p90, models_dir / f"model_{bucket}_p90.joblib")
        logger.info(f"Horizon {bucket}: trained stage1; stage2_p50={stage2_p50 is not None}, stage2_p90={stage2_p90 is not None}")

        # Predictions (long) - include entity_code if present
        cols_to_keep = ["customer_id", "obs_date", "horizon_bucket", "horizon_days", "base_balance_eod"]
        if "entity_code" in g_te.columns:
            cols_to_keep.insert(0, "entity_code")
        outp = g_te[cols_to_keep].copy()
        outp["mco_actual_sgd"] = y_true
        outp["p_runoff"] = p_te
        outp["mco_pred_p50_sgd"] = mco_p50
        outp["mco_pred_p90_sgd"] = mco_p90
        outp["risk_score"] = risk_score
        all_pred_rows.append(outp)

    pred_long = pd.concat(all_pred_rows, ignore_index=True)
    if "entity_code" in pred_long.columns:
        pred_long = pred_long.sort_values(["entity_code", "obs_date", "horizon_days", "customer_id"])
    else:
        pred_long = pred_long.sort_values(["obs_date", "horizon_days", "customer_id"])
    pred_long.to_csv(out_dir / "predictions_long.csv", index=False)

    (out_dir / "metrics_by_horizon.json").write_text(json.dumps(metrics, indent=2))
    return metrics
