#!/usr/bin/env bash
set -euo pipefail

# ============================================
# M3' smoke test (ETL -> train -> infer -> eval)
# usage: ./scripts/smoke_test.sh [configs/default.yaml]
# ============================================

CFG=${1:-configs/default.yaml}
export CFG

if [ ! -f "$CFG" ]; then
  echo "ERROR: config file not found: $CFG" >&2
  exit 1
fi

echo "=== 0) Install deps (Python / NumPy 2系 etc.) ==="
python -V || true
pip -q install -U pip

if [ -f requirements.txt ]; then
  pip -q install -r requirements.txt || true
fi

pip -q install \
  "numpy>=2.0,<2.3" \
  "pandas>=2.0" \
  "scikit-learn>=1.6,<1.7" \
  "matplotlib>=3.8" \
  "scipy>=1.11" \
  "PyYAML>=6.0" \
  "torch>=2.1"

python - <<'PY'
import numpy as np, pandas as pd, sklearn, yaml, sys
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
PY

echo "=== 1) Prepare IO ==="
mkdir -p in out configs

# Jules からの入力アーティファクトを in/ にコピー
if [ -n "${JULES_INPUT_DIR:-}" ]; then
  echo "JULES_INPUT_DIR=${JULES_INPUT_DIR}"
  # DLC CSV
  for f in 1_0_1.csv dlc_result.csv; do
    if [ -f "$JULES_INPUT_DIR/$f" ]; then
      cp -f "$JULES_INPUT_DIR/$f" "in/dlc_result.csv"
      break
    fi
  done
  # transcript
  for f in 1_0_1_transcript.json transcript.json; do
    if [ -f "$JULES_INPUT_DIR/$f" ]; then
      cp -f "$JULES_INPUT_DIR/$f" "in/transcript.json"
      break
    fi
  done
fi

echo "=== 1.1) Read paths from $CFG ==="
eval "$(python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open(os.environ["CFG"], "r", encoding="utf-8"))
paths = cfg["paths"]
for k in ["dlc_csv","pose_timeline","transcript","train_samples","val_samples","out_dir"]:
    v = paths.get(k)
    if v is None:
        continue
    print(f'{k.upper()}="{v}"')
PY
)"

echo "  DLC_CSV       = ${DLC_CSV:-<none>}"
echo "  TRANSCRIPT    = ${TRANSCRIPT:-<none>}"
echo "  POSE_TIMELINE = ${POSE_TIMELINE:-<none>}"
echo "  TRAIN_SAMPLES = ${TRAIN_SAMPLES:-<none>}"
echo "  VAL_SAMPLES   = ${VAL_SAMPLES:-<none>}"
echo "  OUT_DIR       = ${OUT_DIR:-<none>}"

mkdir -p "$(dirname "${POSE_TIMELINE}")" \
         "$(dirname "${TRAIN_SAMPLES}")" \
         "$(dirname "${VAL_SAMPLES}")" \
         "${OUT_DIR}"

echo "=== 2) ETL: DLC CSV -> pose_timeline.json (with mouth6) ==="
if [ ! -f "${DLC_CSV}" ]; then
  echo "ERROR: DLC CSV not found: ${DLC_CSV}" >&2
  exit 1
fi
python -m m3p.etl.dlc_to_pose --config "${CFG}"

echo "=== 3) ETL: transcript + pose_timeline -> train/val JSONL ==="
if [ ! -f "${TRANSCRIPT}" ]; then
  echo "ERROR: transcript.json not found: ${TRANSCRIPT}" >&2
  exit 1
fi
python -m m3p.etl.build_trainset --config "${CFG}"

echo "=== 4) Train Text->mouth6 model ==="
python -m m3p.train --config "${CFG}"

echo "=== 5) Infer on VAL set (JSONL) ==="
PRED_SAMPLES="${OUT_DIR}/pred.val.jsonl"
python -m m3p.infer.jsonl_infer \
  --config "${CFG}" \
  --in  "${VAL_SAMPLES}" \
  --out "${PRED_SAMPLES}"

echo "=== 6) Eval ==="
REPORT_PATH="${OUT_DIR}/report.json"
python -m m3p.eval \
  --config "${CFG}" \
  --gt    "${VAL_SAMPLES}" \
  --pred  "${PRED_SAMPLES}" \
  --report "${REPORT_PATH}"

echo "=== 7) Summary ==="
python - <<PY
import json, os, yaml
cfg = yaml.safe_load(open("${CFG}", "r", encoding="utf-8"))
paths = cfg["paths"]
out_dir = paths["out_dir"]
report_path = os.path.join(out_dir, "report.json")
print("out_dir:", out_dir)
if os.path.exists(report_path):
    rep = json.load(open(report_path, "r", encoding="utf-8"))
    print("Eval report:", report_path)
    for k, v in rep.items():
        print(f"  {k}: {v}")
else:
    print("WARNING: report.json not found.")
PY

echo "=== DONE (M3' smoke test) ==="
