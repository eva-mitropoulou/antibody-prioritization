from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text())


def csv_columns(relative_path: str) -> list[str]:
    with (ROOT / relative_path).open(newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def test_required_final_reports_exist() -> None:
    required = [
        "README.md",
        "docs/DATA_CARD.md",
        "docs/MODEL_CARD.md",
        "reports/final_project_report.md",
        "reports/final_artifact_map.md",
        "reports/final_consistency_audit.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing


def test_required_metrics_exist() -> None:
    required = [
        "reports/metrics/model_registry.json",
        "reports/metrics/matched_kmer_benchmark_audit.json",
        "reports/metrics/source_holdout_validation_metrics.json",
        "reports/metrics/source_robust_model_selection_metrics.json",
        "reports/metrics/calibration_threshold_metrics.json",
        "reports/metrics/oas_background_retrieval_metrics.json",
        "reports/metrics/final_consistency_audit.json",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing


def test_public_score_csv_headers_do_not_expose_raw_sequence_columns() -> None:
    score_csvs = [
        "reports/oas_background_retrieval_scores.csv",
        "reports/oas_matched_background_retrieval_scores.csv",
        "reports/active_learning_selected_records.csv",
        "reports/model_error_analysis_predictions.csv",
    ]
    forbidden_exact = {
        "sequence",
        "heavy_sequence",
        "light_sequence",
        "sequence_pair_text",
        "vhorvhh",
        "vl",
    }
    for path in score_csvs:
        if not (ROOT / path).exists():
            continue
        normalized_columns = {column.strip().lower() for column in csv_columns(path)}
        assert not (normalized_columns & forbidden_exact), path


def test_oas_score_csv_uses_ids_not_raw_sequences() -> None:
    oas_csvs = [
        "reports/oas_background_retrieval_scores.csv",
        "reports/oas_matched_background_retrieval_scores.csv",
    ]
    allowed_id_columns = {
        "sequence_pair_hash",
        "hashed_sequence_key",
        "record_id",
        "row_id",
    }
    forbidden_exact = {
        "sequence",
        "heavy_sequence",
        "light_sequence",
        "sequence_pair_text",
        "vhorvhh",
        "vl",
    }
    for path in oas_csvs:
        columns = {column.strip().lower() for column in csv_columns(path)}
        assert columns & allowed_id_columns, path
        assert not (columns & forbidden_exact), path


def test_final_report_oas_wording() -> None:
    text = (ROOT / "reports/final_project_report.md").read_text().lower()
    assert "unknown-target background" in text
    assert "assayed negative-class" in text


def test_source_robust_selected_model() -> None:
    metrics = read_json("reports/metrics/source_robust_model_selection_metrics.json")
    assert metrics["model_selection"]["selected_model"] == "whole_pair_kmer"


def test_threshold_07_appears_in_calibration_outputs() -> None:
    calibration = read_json("reports/metrics/calibration_threshold_metrics.json")
    source_robust = read_json("reports/metrics/source_robust_model_selection_metrics.json")
    thresholds = {
        round(float(row["threshold"]), 1)
        for row in calibration.get("threshold_metrics", [])
        if row.get("threshold") is not None
    }
    selected = source_robust["model_selection"]["selected_model"]
    robust_threshold = source_robust["calibration_results"][selected][
        "best_high_confidence_threshold"
    ]["threshold"]
    thresholds.add(round(float(robust_threshold), 1))
    assert 0.7 in thresholds


def test_source_holdout_overlap_zero_where_reported() -> None:
    metrics = read_json("reports/metrics/source_holdout_validation_metrics.json")
    overlap_values: list[int] = []

    def collect(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "group_overlap_count" and value is not None:
                    overlap_values.append(int(value))
                else:
                    collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(metrics)
    assert overlap_values
    assert all(value == 0 for value in overlap_values)
