# mco/config.py
from __future__ import annotations

import logging

LOG_LEVEL: str = "INFO"

BASE_CCY: str = "SGD"

# Conservative tail knobs
P90_PROB_FLOOR: float = 0.80
RISK_SCORE_POWER: float = 2.0

# Runoff event definition (stage 1 label)
RUNOFF_ABS_THRESHOLD: float = 2000.0
RUNOFF_REL_THRESHOLD: float = 0.05  # 5% of balance

# Rolling features
ROLLING_WINDOW_DAYS: int = 30
ROLLING_MIN_PERIODS: int = 10

# Split controls
TEST_SIZE: float = 0.30
RANDOM_STATE: int = 42

# Model hyperparams
LOGREG_SOLVER: str = "liblinear"
LOGREG_MAX_ITER: int = 200
RIDGE_ALPHA: float = 1.0
QREG_QUANTILE: float = 0.90
QREG_ALPHA: float = 0.0
QREG_SOLVER: str = "highs"

# -----------------------------
# Multi-horizon config
# -----------------------------
# Full Treasury reporting horizons (actual labels)
HORIZON_BUCKETS_REPORT = {
    "ON": 1,
    "D2": 2,
    "D3": 3,
    "D4": 4,
    "D5": 5,
    "D6": 6,
    "W1": 7,
    "W2": 14,
    "M1": 30,
    "M3": 90,
    "M6": 180,
    "M12": 365,
    "TOTAL": 365,
}

# Horizons to train ML models on (expanded for short-term and long-term forecasting)
HORIZON_BUCKETS_TRAIN = {
    "ON": 1,
    "D2": 2,
    "D3": 3,
    "D4": 4,
    "D5": 5,
    "D6": 6,
    "W1": 7,
    "W2": 14,
    "M1": 30,
    "M3": 90,
    "M6": 180,
    "M12": 365,
    "TOTAL": 365,
}

# ---- Reporting & UI Magic Numbers ----
DEFAULT_DAYS_SINCE_SALARY_MISSING: int = 999
MIN_RUNOFF_CASES_FOR_STAGE2: int = 20
TOP_CONTRIBUTORS_N: int = 30
TAIL_CAPTURE_PCT: float = 0.05
