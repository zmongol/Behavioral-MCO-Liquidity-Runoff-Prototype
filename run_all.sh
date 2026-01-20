#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-big}"   # small | big | custom
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- Activate venv ---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "[WARNING] .venv not found (running without virtual environment)"
fi

ZIP=""
DATA_TAG=""

# --- Spinner / Progress utility ---
# Shows a spinner with elapsed seconds while a command runs
run_with_spinner() {
  local label="$1"; shift
  # Print initial label
  printf "%s\n" "$label"
  (
    set -e
    "$@"
  ) &
  local pid=$!
  local sp='|/-\\'
  local i=0
  local start
  start=$(date +%s)
  # Hide cursor if possible
  tput civis 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do
    local now
    now=$(date +%s)
    local elapsed=$(( now - start ))
    printf "\r%s %c  %ds elapsed" "$label" "${sp:i++%${#sp}:1}" "$elapsed"
    sleep 0.2
  done
  # Wait to capture exit code
  wait "$pid"
  local status=$?
  # Restore cursor
  tput cnorm 2>/dev/null || true
  if [[ $status -eq 0 ]]; then
    local end
    end=$(date +%s)
    local total=$(( end - start ))
    printf "\r%s Ã¢Å“â€œ  %ds total\n" "$label" "$total"
  else
    printf "\r%s Ã¢Å“â€” (exit %d)\n" "$label" "$status"
    return $status
  fi
}

if [[ "$MODE" == "small" ]]; then
  ZIP="mco_mock_data.zip"
  DATA_TAG="small"
elif [[ "$MODE" == "big" ]]; then
  ZIP="mco_mock_data_big.zip"
  DATA_TAG="big"
  # Step 0: Generate dataset (label derived from generator constants)
  LABEL=$(python - <<'PY'
import importlib.util, pathlib
fp = pathlib.Path('generate_big_dataset.py')
spec = importlib.util.spec_from_file_location('gen', fp)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
customers = getattr(mod, 'TARGET_CUSTOMERS', 'n/a')
years = getattr(mod, 'YEARS_OF_HISTORY', 'n/a')
days = getattr(mod, 'DAYS_OF_HISTORY', 'n/a')
print(f"[0/3] Generating dataset ({years}yr, {customers:,} customers, {days} days)...")
PY
)
  run_with_spinner "$LABEL" python generate_big_dataset.py --out "$ZIP"
elif [[ "$MODE" == "custom" ]]; then
  ZIP="${2:-}"
  DATA_TAG="custom"
  if [[ -z "$ZIP" ]]; then
    echo "[ERROR] custom mode requires <zipfile>"
    echo "Example: ./run_all.sh custom mco_mock_data_big.zip"
    exit 1
  fi
else
  echo "[ERROR] Unknown mode: $MODE (use small|big|custom)"
  exit 1
fi

if [[ ! -f "$ZIP" ]]; then
  echo "[ERROR] ZIP file not found: $ROOT_DIR/$ZIP"
  ls -1 *.zip 2>/dev/null || true
  exit 1
fi

# --- Timestamped run folder ---
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${TS}_${DATA_TAG}"
OUT_DIR="${RUN_DIR}/outputs"
RPT_DIR="${RUN_DIR}/reports"
META_DIR="${RUN_DIR}/metadata"

mkdir -p "$OUT_DIR" "$RPT_DIR" "$META_DIR"

echo "=========================================="
echo "MCO OC Runner (timestamped)"
echo "Mode     : $MODE"
echo "ZIP      : $ZIP"
echo "RUN_DIR  : $RUN_DIR"
echo "Python   : $(python --version)"
echo "=========================================="

# --- Save metadata snapshot (simple) ---
python - <<PY
import json, os, sys, platform, datetime
meta = {
  "run_ts": "${TS}",
  "mode": "${MODE}",
  "zip": "${ZIP}",
  "python": sys.version,
  "platform": platform.platform(),
}
os.makedirs("${META_DIR}", exist_ok=True)
with open(os.path.join("${META_DIR}", "run_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote:", os.path.join("${META_DIR}", "run_metadata.json"))
PY

# --- Train ---
echo ""
if [[ "$MODE" == "big" ]]; then
  run_with_spinner "[1/3] Training model..." python run_train.py --zip "$ZIP" --out "$OUT_DIR" --mode train
else
  run_with_spinner "[1/3] Training & reporting (small)..." python run_train.py --zip "$ZIP" --out "$OUT_DIR" --mode report
fi

# --- Report pack (existing) ---
echo ""
run_with_spinner "[2/3] Building base reports..." python make_mco_report.py \
  --labels "$OUT_DIR/mco_labels_long.csv" \
  --pred "$OUT_DIR/predictions_long.csv" \
  --out "$RPT_DIR"



# --- Tail capture quick summary ---
echo ""
echo "[3/3] Tail capture summary..."

tail_summary() {
  python - <<PY
import pandas as pd, os
path = os.path.join("${RPT_DIR}", "tail_capture_by_horizon_overall.csv")
df = pd.read_csv(path)
print("tail_capture_by_horizon_overall.csv =", path)
print("avg =", df["tail_capture_rate"].mean(),
      "min =", df["tail_capture_rate"].min(),
      "max =", df["tail_capture_rate"].max())
PY
}

run_with_spinner "[3/3] Summarizing tail capture..." tail_summary

echo ""
echo "Ã¢Å“â€¦ Done."
echo "Run folder:"
echo "  $ROOT_DIR/$RUN_DIR"
