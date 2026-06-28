"""Audit the raw CoV-AbDab table before any modeling work.

Run from the project root:

    python src/data/audit_covabdab.py
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# Resolve paths relative to this file so the script works from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "covabdab_audit.md"

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx"}

EMPTY_STRINGS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "nd",
    "n d",
    "none",
    "null",
    "not applicable",
    "not available",
    "not determined",
    "not done",
    "not reported",
    "not sequenced",
    "unknown",
    "unavailable",
}


FIELD_KEYWORDS = {
    "antibody_name": [
        "name",
        "antibody name",
        "antibody",
        "ab name",
        "clone",
    ],
    "heavy_chain_sequence": [
        "vh or vhh",
        "heavy chain sequence",
        "heavy sequence",
        "vh sequence",
        "vhh sequence",
        "variable heavy",
        "heavy variable",
        "vh",
        "vhh",
    ],
    "light_chain_sequence": [
        "vl",
        "light chain sequence",
        "light sequence",
        "variable light",
        "light variable",
        "kappa",
        "lambda",
    ],
    "bind_target_antigen": [
        "binds to",
        "does not bind",
        "doesn't bind",
        "binding",
        "target",
        "antigen",
        "virus",
        "protein",
    ],
    "sars_cov_2": [
        "sars-cov-2",
        "sars cov 2",
        "sars_cov_2",
        "sarscov2",
        "covid-19",
        "2019-ncov",
    ],
    "neutralisation": [
        "neutralising",
        "neutralizing",
        "neutralisation",
        "neutralization",
        "neut",
    ],
    "pdb_or_structure": [
        "pdb",
        "structure",
        "structures",
    ],
    "epitope": [
        "epitope",
        "protein epitope",
        "protein + epitope",
        "binding site",
    ],
}


# These terms help avoid confusing CDRs, V genes, or structures with full
# heavy/light-chain sequence columns.
SEQUENCE_EXCLUDES = {
    "cdr",
    "gene",
    "germline",
    "model",
    "pdb",
    "source",
    "structure",
    "date",
}

SARS_COV_2_PATTERN = re.compile(
    r"sars[\s_-]*cov[\s_-]*2|sarscov2|2019[\s_-]*ncov|covid[\s_-]*19",
    flags=re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    """Lowercase text and collapse punctuation to spaces for matching."""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_nonempty(value: object) -> bool:
    """Treat common placeholder strings as missing values."""
    if pd.isna(value):
        return False

    text = str(value).strip()
    if normalize_text(text) in EMPTY_STRINGS:
        return False

    return True


def nonempty_mask(series: pd.Series) -> pd.Series:
    """Return True for cells that contain useful information."""
    return series.apply(is_nonempty)


def find_data_file(raw_dir: Path) -> Path:
    """Find the first supported raw data file, preferring CoV-AbDab names."""
    candidates = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported data file found in {raw_dir}. "
            f"Expected one of: {supported}"
        )

    def sort_key(path: Path) -> tuple[int, str]:
        has_covabdab_name = "covabdab" in path.name.lower() or "cov-abdab" in path.name.lower()
        return (0 if has_covabdab_name else 1, path.name.lower())

    return sorted(candidates, key=sort_key)[0]


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV, TSV, or XLSX files using the extension as the format hint."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix == ".xlsx":
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx files requires pandas' optional Excel engine "
                "(usually openpyxl). Use a CSV/TSV file or install openpyxl."
            ) from exc

    raise ValueError(f"Unsupported file type: {path.suffix}")


def column_match_score(column: str, keywords: list[str]) -> float:
    """Score how well a column name matches a list of expected concepts."""
    normalized_column = normalize_text(column)
    column_tokens = set(normalized_column.split())
    best_score = 0.0

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        keyword_tokens = set(normalized_keyword.split())

        if normalized_column == normalized_keyword:
            score = 1.0
        elif normalized_keyword in normalized_column:
            score = 0.90
        elif normalized_column in normalized_keyword:
            score = 0.80
        else:
            overlap = len(column_tokens & keyword_tokens) / max(len(keyword_tokens), 1)
            similarity = SequenceMatcher(None, normalized_column, normalized_keyword).ratio()
            score = max(overlap * 0.75, similarity * 0.65)

        best_score = max(best_score, score)

    return best_score


def detect_columns(
    columns: list[str],
    keywords: list[str],
    threshold: float = 0.45,
    exclude_terms: set[str] | None = None,
) -> list[str]:
    """Return likely matching columns sorted by match quality."""
    exclude_terms = exclude_terms or set()
    matches: list[tuple[float, str]] = []

    for column in columns:
        normalized_column = normalize_text(column)
        if any(term in normalized_column.split() for term in exclude_terms):
            continue

        score = column_match_score(column, keywords)
        if score >= threshold:
            matches.append((score, column))

    return [column for _, column in sorted(matches, key=lambda item: (-item[0], item[1]))]


def columns_containing_pattern(
    df: pd.DataFrame,
    pattern: re.Pattern[str],
    columns: list[str] | None = None,
) -> list[str]:
    """Find text columns where at least one value matches a regex pattern."""
    matches: list[str] = []
    columns_to_scan = columns if columns is not None else list(df.columns)

    for column in columns_to_scan:
        is_text_column = (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        )
        if not is_text_column:
            continue

        contains_pattern = df[column].fillna("").astype(str).str.contains(pattern, regex=True)
        if bool(contains_pattern.any()):
            matches.append(column)

    return matches


def detect_all_fields(df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify likely columns for each concept we need in the audit."""
    columns = list(df.columns)
    detected: dict[str, list[str]] = {}

    for field_name, keywords in FIELD_KEYWORDS.items():
        exclude_terms = SEQUENCE_EXCLUDES if "chain_sequence" in field_name else set()
        detected[field_name] = detect_columns(columns, keywords, exclude_terms=exclude_terms)

    # SARS-CoV-2 is usually encoded as values such as "SARS-CoV2_WT", not as a
    # dedicated column name. Scan target/assay-like columns first so literature
    # notes do not inflate the antigen-label count.
    sars_scan_candidates = [
        column
        for column in columns
        if column
        in set(
            detected["bind_target_antigen"]
            + detected["neutralisation"]
            + detected["epitope"]
        )
    ]
    sars_value_columns = columns_containing_pattern(
        df,
        SARS_COV_2_PATTERN,
        columns=sars_scan_candidates,
    )
    if not sars_value_columns:
        sars_value_columns = columns_containing_pattern(df, SARS_COV_2_PATTERN)

    if sars_value_columns:
        detected["sars_cov_2"] = [
            column
            for column in columns
            if column in set(detected["sars_cov_2"] + sars_value_columns)
        ]

    return detected


def count_rows_with_any_value(df: pd.DataFrame, columns: list[str]) -> int:
    """Count rows where at least one selected column is non-empty."""
    if not columns:
        return 0

    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask = mask | nonempty_mask(df[column])

    return int(mask.sum())


def count_sars_cov_2_rows(df: pd.DataFrame, detected: dict[str, list[str]]) -> int:
    """Count rows that mention SARS-CoV-2 in target-like columns."""
    likely_columns = []
    for field_name in ("sars_cov_2", "bind_target_antigen", "neutralisation", "epitope"):
        likely_columns.extend(detected.get(field_name, []))

    # If the column names do not give us enough signal, search all text columns.
    if not likely_columns:
        likely_columns = [
            column
            for column in df.columns
            if pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ]

    if not likely_columns:
        return 0

    row_text = df[sorted(set(likely_columns))].fillna("").astype(str).agg(" ".join, axis=1)
    return int(row_text.str.contains(SARS_COV_2_PATTERN, regex=True).sum())


def split_neutralisation_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split neutralisation columns into positive and negative label columns."""
    positive_columns: list[str] = []
    negative_columns: list[str] = []

    negative_markers = {"not", "non", "doesn", "doesnt", "lack", "negative"}

    for column in columns:
        normalized = normalize_text(column)
        tokens = set(normalized.split())

        if tokens & negative_markers:
            negative_columns.append(column)
        else:
            positive_columns.append(column)

    return positive_columns, negative_columns


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing and non-missing values with our placeholder handling."""
    rows = []
    total_rows = len(df)

    for column in df.columns:
        non_missing = int(nonempty_mask(df[column]).sum())
        missing = total_rows - non_missing
        missing_percent = (missing / total_rows * 100) if total_rows else 0.0
        rows.append(
            {
                "column": column,
                "missing": missing,
                "non_missing": non_missing,
                "missing_percent": round(missing_percent, 2),
            }
        )

    return pd.DataFrame(rows)


def values_preview(df: pd.DataFrame, columns: list[str], limit: int = 8) -> dict[str, list[str]]:
    """Collect a short preview of values from selected columns for the report."""
    preview: dict[str, list[str]] = {}

    for column in columns:
        values = (
            df.loc[nonempty_mask(df[column]), column]
            .astype(str)
            .drop_duplicates()
            .head(limit)
            .tolist()
        )
        preview[column] = values

    return preview


def format_list(values: list[str]) -> str:
    """Format a list for Markdown table cells."""
    return ", ".join(values) if values else "Not detected"


def build_report(
    df: pd.DataFrame,
    source_file: Path,
    detected: dict[str, list[str]],
    counts: dict[str, int],
    missing_summary: pd.DataFrame,
) -> str:
    """Build the Markdown audit report."""
    lines: list[str] = []

    lines.append("# CoV-AbDab Data Audit")
    lines.append("")
    lines.append(f"Source file: `{source_file.relative_to(PROJECT_ROOT)}`")
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    lines.append("## Key counts")
    lines.append("")
    lines.append(f"- Number of rows: {counts['rows']}")
    lines.append(f"- Number of columns: {counts['columns']}")
    lines.append(f"- Rows marked SARS-CoV-2: {counts['sars_cov_2_rows']}")
    lines.append(f"- Rows with heavy-chain sequence: {counts['heavy_sequence_rows']}")
    lines.append(f"- Rows with light-chain sequence: {counts['light_sequence_rows']}")
    lines.append(f"- Rows with both heavy and light-chain sequence: {counts['both_sequence_rows']}")
    lines.append(f"- Rows marked neutralising: {counts['neutralising_rows']}")
    lines.append(f"- Rows marked non-neutralising: {counts['non_neutralising_rows']}")
    lines.append("")

    lines.append("## Column names")
    lines.append("")
    for column in df.columns:
        lines.append(f"- {column}")
    lines.append("")

    lines.append("## Detected label and sequence columns")
    lines.append("")
    lines.append("| Concept | Likely columns | Rows with relevant values |")
    lines.append("|---|---|---:|")
    for field_name, columns in detected.items():
        if field_name == "sars_cov_2":
            non_missing = counts["sars_cov_2_rows"]
        else:
            non_missing = count_rows_with_any_value(df, columns)
        concept = field_name.replace("_", " ")
        lines.append(f"| {concept} | {format_list(columns)} | {non_missing} |")
    lines.append("")

    lines.append("## Available labels")
    lines.append("")
    label_fields = {
        "Binding target / antigen": detected.get("bind_target_antigen", []),
        "Neutralisation": detected.get("neutralisation", []),
        "Target / epitope": detected.get("epitope", []),
        "PDB / structure": detected.get("pdb_or_structure", []),
    }
    for label_name, columns in label_fields.items():
        status = "available" if columns else "not detected"
        lines.append(f"- {label_name}: {status} ({format_list(columns)})")
    lines.append("")

    lines.append("## Example values from detected label columns")
    lines.append("")
    preview_columns = sorted(
        set(
            detected.get("bind_target_antigen", [])
            + detected.get("neutralisation", [])
            + detected.get("epitope", [])
            + detected.get("pdb_or_structure", [])
        )
    )
    preview = values_preview(df, preview_columns)
    for column, values in preview.items():
        lines.append(f"### {column}")
        lines.append("")
        if values:
            for value in values:
                lines.append(f"- {value}")
        else:
            lines.append("- No non-empty values found")
        lines.append("")

    lines.append("## Missing values per column")
    lines.append("")
    lines.append("| Column | Missing | Non-missing | Missing % |")
    lines.append("|---|---:|---:|---:|")
    for row in missing_summary.itertuples(index=False):
        lines.append(
            f"| {row.column} | {row.missing} | {row.non_missing} | {row.missing_percent:.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run the CoV-AbDab audit and write the Markdown report."""
    data_file = find_data_file(RAW_DIR)
    df = load_table(data_file)

    detected = detect_all_fields(df)

    heavy_columns = detected.get("heavy_chain_sequence", [])
    light_columns = detected.get("light_chain_sequence", [])
    neutralisation_columns = detected.get("neutralisation", [])
    neutralising_columns, non_neutralising_columns = split_neutralisation_columns(
        neutralisation_columns
    )

    heavy_mask = pd.Series(False, index=df.index)
    for column in heavy_columns:
        heavy_mask = heavy_mask | nonempty_mask(df[column])

    light_mask = pd.Series(False, index=df.index)
    for column in light_columns:
        light_mask = light_mask | nonempty_mask(df[column])

    counts = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sars_cov_2_rows": count_sars_cov_2_rows(df, detected),
        "heavy_sequence_rows": int(heavy_mask.sum()),
        "light_sequence_rows": int(light_mask.sum()),
        "both_sequence_rows": int((heavy_mask & light_mask).sum()),
        "neutralising_rows": count_rows_with_any_value(df, neutralising_columns),
        "non_neutralising_rows": count_rows_with_any_value(df, non_neutralising_columns),
    }

    missing_summary = missing_value_summary(df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(df, data_file, detected, counts, missing_summary),
        encoding="utf-8",
    )

    print(f"Loaded file: {data_file}")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nKey counts:")
    print(f"- Rows marked SARS-CoV-2: {counts['sars_cov_2_rows']}")
    print(f"- Rows with heavy-chain sequence: {counts['heavy_sequence_rows']}")
    print(f"- Rows with light-chain sequence: {counts['light_sequence_rows']}")
    print(f"- Rows with both heavy and light-chain sequence: {counts['both_sequence_rows']}")
    print(f"- Rows marked neutralising: {counts['neutralising_rows']}")
    print(f"- Rows marked non-neutralising: {counts['non_neutralising_rows']}")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
