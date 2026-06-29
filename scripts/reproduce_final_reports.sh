#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

FORCE_REBUILD="${FORCE_REBUILD:-0}"
RUN_TESTS="${RUN_TESTS:-1}"

run_python_if_present() {
  local label="$1"
  local script="$2"
  if [[ -f "$script" ]]; then
    echo "[run] $label"
    "$PY" "$script" || echo "[warn] $label failed"
  else
    echo "[skip] $label missing: $script"
  fi
}

run_if_outputs_missing() {
  local label="$1"
  local script="$2"
  shift 2
  local missing=0
  for output in "$@"; do
    if [[ ! -s "$output" ]]; then
      missing=1
    fi
  done

  if [[ "$FORCE_REBUILD" == "1" || "$missing" == "1" ]]; then
    run_python_if_present "$label" "$script"
  else
    echo "[skip] $label artifacts already exist"
  fi
}

echo "[info] project root: $ROOT"
echo "[info] python: $PY"

run_if_outputs_missing \
  "source holdout and calibration" \
  "src/analysis/run_source_holdout_and_calibration.py" \
  "reports/metrics/source_holdout_validation_metrics.json" \
  "reports/metrics/calibration_threshold_metrics.json"

run_if_outputs_missing \
  "source robust model selection" \
  "src/analysis/run_source_robust_model_selection.py" \
  "reports/metrics/source_robust_model_selection_metrics.json" \
  "reports/source_robust_model_comparison.csv" \
  "reports/source_holdout_failure_analysis.csv"

if [[ -s "data/processed/oas/oas_paired_standardized.csv" ]]; then
  run_if_outputs_missing \
    "OAS background retrieval" \
    "src/analysis/run_oas_background_retrieval.py" \
    "reports/metrics/oas_background_retrieval_metrics.json" \
    "reports/oas_background_retrieval_scores.csv"

  run_if_outputs_missing \
    "matched OAS background retrieval" \
    "src/analysis/run_oas_matched_background_retrieval.py" \
    "reports/metrics/oas_matched_background_retrieval_metrics.json" \
    "reports/oas_matched_background_retrieval_scores.csv"
else
  echo "[skip] OAS retrieval skipped because local standardized OAS data is missing"
fi

run_python_if_present "final project report generation" "src/analysis/build_final_project_report.py"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[run] pytest"
  "$PY" -m pytest -q || echo "[warn] pytest failed"
else
  echo "[skip] pytest disabled with RUN_TESTS=0"
fi
