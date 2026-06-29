"""Build final artifact map and consistency audit without inspecting raw records."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
ARTIFACT_MAP_PATH = REPORTS_DIR / "final_artifact_map.md"
AUDIT_REPORT_PATH = REPORTS_DIR / "final_consistency_audit.md"
AUDIT_JSON_PATH = METRICS_DIR / "final_consistency_audit.json"


EXPECTED_ARTIFACTS = [
    "README.md",
    "docs/DATA_CARD.md",
    "docs/MODEL_CARD.md",
    "scripts/reproduce_final_reports.sh",
    "tests/test_project_integrity.py",
    "reports/final_project_report.md",
    "reports/model_registry.md",
    "reports/source_robust_model_selection_report.md",
    "reports/calibration_threshold_report.md",
    "reports/oas_background_retrieval_report.md",
    "reports/oas_existing_record_shortlist_report.md",
    "reports/matched_kmer_benchmark_audit.md",
    "reports/unsupervised_antibody_landscape_report.md",
    "reports/active_learning_simulation_report.md",
    "reports/metrics/model_registry.json",
    "reports/metrics/source_robust_model_selection_metrics.json",
    "reports/metrics/source_holdout_validation_metrics.json",
    "reports/metrics/calibration_threshold_metrics.json",
    "reports/metrics/oas_background_retrieval_metrics.json",
    "reports/metrics/oas_existing_record_shortlist_metrics.json",
    "reports/metrics/matched_kmer_benchmark_audit.json",
    "reports/oas_existing_record_shortlist_top25.csv",
    "reports/oas_existing_record_shortlist_top100.csv",
    "reports/oas_existing_record_scores_public.csv",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sorted_existing(pattern: str) -> list[str]:
    return sorted(rel(path) for path in PROJECT_ROOT.glob(pattern) if path.is_file())


def tracked_files() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"], cwd=PROJECT_ROOT, text=True
        )
    except Exception:
        return []
    return sorted(path for path in output.splitlines() if path)


def load_json(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_text(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def csv_header(relative_path: str) -> list[str]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return next(csv.reader(handle), [])


def build_artifact_map() -> str:
    tracked = tracked_files()

    def pick(prefix: str, suffix: str | None = None) -> list[str]:
        paths = [path for path in tracked if path.startswith(prefix)]
        if suffix is not None:
            paths = [path for path in paths if path.endswith(suffix)]
        return paths

    sections = {
        "Project Docs": [
            path
            for path in tracked
            if path in {"README.md", "data/README.md"}
            or path.startswith("docs/")
        ],
        "Model Artifacts": pick("models/"),
        "Metrics Artifacts": pick("reports/metrics/", ".json"),
        "Report Artifacts": pick("reports/", ".md"),
        "Figure Artifacts": pick("reports/figures/"),
        "Sanitized CSV Outputs": pick("reports/", ".csv"),
        "Scripts": pick("scripts/") + pick("src/", ".py"),
        "Tests": pick("tests/", ".py"),
    }
    lines = [
        "# Artifact Index",
        "",
        "This map lists committed project artifacts by path. Local raw and processed sequence tables stay outside the public repository.",
        "",
    ]
    for title, paths in sections.items():
        lines.extend([f"## {title}", ""])
        if paths:
            lines.extend(f"- `{path}`" for path in paths)
        else:
            lines.append("- None found")
        lines.append("")
    return "\n".join(lines)


def selected_models() -> dict[str, Any]:
    registry = load_json("reports/metrics/model_registry.json")
    source_robust = load_json("reports/metrics/source_robust_model_selection_metrics.json")
    primary = registry.get("primary_broad_scorer", {})
    return {
        "model_registry_primary_model_id": primary.get("model_id"),
        "model_registry_primary_family": primary.get("family"),
        "source_robust_selected_model": source_robust.get("model_selection", {}).get("selected_model"),
    }


def public_score_header_check() -> dict[str, Any]:
    score_csvs = [
        "reports/oas_background_retrieval_scores.csv",
        "reports/oas_matched_background_retrieval_scores.csv",
        "reports/oas_existing_record_shortlist_top25.csv",
        "reports/oas_existing_record_shortlist_top100.csv",
        "reports/oas_existing_record_scores_public.csv",
        "reports/source_robust_model_comparison.csv",
        "reports/source_holdout_failure_analysis.csv",
    ]
    forbidden_exact = {
        "sequence",
        "heavy_sequence",
        "light_sequence",
        "sequence_pair_text",
        "vhorvhh",
        "vl",
    }
    results = {}
    for path in score_csvs:
        columns = {column.strip().lower() for column in csv_header(path)}
        results[path] = {
            "exists": (PROJECT_ROOT / path).exists(),
            "forbidden_columns_found": sorted(columns & forbidden_exact),
        }
    return results


def build_audit() -> tuple[str, dict[str, Any]]:
    final_report = read_text("reports/final_project_report.md").lower()
    readme = read_text("README.md").lower()
    data_card = read_text("docs/DATA_CARD.md").lower()
    model_card = read_text("docs/MODEL_CARD.md").lower()
    oas_shortlist_report = read_text(
        "reports/oas_existing_record_shortlist_report.md"
    ).lower()
    combined_project_text = "\n".join(
        [final_report, readme, data_card, model_card, oas_shortlist_report]
    )
    models = selected_models()
    missing_expected = [
        path for path in EXPECTED_ARTIFACTS if not (PROJECT_ROOT / path).is_file()
    ]
    dangerous_affirmative_phrases = [
        "proves therapeutic efficacy",
        "demonstrates therapeutic efficacy",
        "production-ready therapeutic",
        "prospective therapeutic prediction is supported",
        "prospective antibody design",
        "generates optimized antibodies",
    ]
    header_checks = public_score_header_check()
    checks = {
        "expected_artifacts_exist": not missing_expected,
        "selected_model_consistent": (
            models["source_robust_selected_model"] == "whole_pair_kmer"
            and models["model_registry_primary_model_id"] == "kmer_tfidf_logreg_pair_text"
        ),
        "oas_described_as_unknown_target_background": "unknown-target background" in combined_project_text,
        "oas_existing_record_shortlist_documented": (
            "oas existing-record retrieval shortlist" in combined_project_text
            and "existing-record shortlist" in combined_project_text
        ),
        "oas_shortlist_score_not_binding_probability": (
            "not a binding probability" in combined_project_text
        ),
        "oas_shortlist_not_sequence_generation": (
            "does not generate or modify sequences" in combined_project_text
            or "does not generate, mutate, design, optimize, or propose sequences"
            in combined_project_text
        ),
        "oas_not_described_as_assayed_negative_class": "assayed negative-class" in combined_project_text,
        "source_holdout_limitations_included": (
            "source-holdout" in combined_project_text
            and "source/study effects" in combined_project_text
        ),
        "no_affirmative_prospective_or_efficacy_overclaim": not any(
            phrase in combined_project_text for phrase in dangerous_affirmative_phrases
        ),
        "public_score_csv_headers_safe": all(
            not result["forbidden_columns_found"] for result in header_checks.values()
        ),
    }
    status = "PASS" if all(checks.values()) else "WARN"
    audit_json = {
        "status": status,
        "checks": checks,
        "selected_models": models,
        "missing_expected_artifacts": missing_expected,
        "public_score_header_checks": header_checks,
        "next_recommended_action": (
            "Repository project files are internally consistent."
            if status == "PASS"
            else "Inspect failed checks and regenerate project artifacts."
        ),
    }
    lines = [
        "# Final Consistency Audit",
        "",
        f"Overall status: **{status}**",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Selected Models",
            "",
            f"- Model registry primary model: `{models['model_registry_primary_model_id']}`",
            f"- Source-robust selected model: `{models['source_robust_selected_model']}`",
            "",
            "## Missing Expected Artifacts",
            "",
        ]
    )
    if missing_expected:
        lines.extend(f"- `{path}`" for path in missing_expected)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OAS is treated as unknown-target background and kept separate from the neutralisation benchmark.",
            "- The OAS existing-record retrieval shortlist is tracked as expert-review prioritization, not antibody design.",
            "- Source-holdout limitations are preserved in project documentation.",
            "- The audit checks paths, report wording, JSON summaries, and CSV headers only.",
            "",
        ]
    )
    return "\n".join(lines), audit_json


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_MAP_PATH.write_text(build_artifact_map())
    audit_report, audit_json = build_audit()
    AUDIT_REPORT_PATH.write_text(audit_report)
    AUDIT_JSON_PATH.write_text(json.dumps(audit_json, indent=2, sort_keys=True) + "\n")
    print("Final consistency audit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
