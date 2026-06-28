"""Build an unsupervised antibody sequence landscape."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _safe_analysis_utils import (
    PROJECT_ROOT,
    compact_pair_from_columns,
    label_counts_dict,
    label_series,
    light_status_series,
    read_csv_text,
    safe_output_columns,
    target_region_group_series,
    value_counts_dict,
    write_json,
    write_text,
)


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
OUTPUT_CSV_PATH = PROJECT_ROOT / "reports" / "unsupervised_antibody_clusters.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "unsupervised_antibody_landscape_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "unsupervised_antibody_landscape_metrics.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "unsupervised_antibody_landscape.png"

RANDOM_STATE = 42


def load_features(data: pd.DataFrame) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Load cached embeddings or build compact k-mer SVD features."""
    if PAIR_EMBEDDING_PATH.exists():
        embeddings = np.load(PAIR_EMBEDDING_PATH)
        if embeddings.shape[0] == len(data):
            return embeddings.astype(float), "cached_pair_embeddings", {
                "path": str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
                "shape": list(embeddings.shape),
            }
    compact = compact_pair_from_columns(data)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=6000)
    matrix = vectorizer.fit_transform(compact)
    n_components = min(50, max(2, matrix.shape[1] - 1), max(2, matrix.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    features = svd.fit_transform(matrix)
    return features, "compact_kmer_svd_fallback", {
        "tfidf_features": int(matrix.shape[1]),
        "svd_components": int(n_components),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }


def reduce_for_plot(features: np.ndarray) -> np.ndarray:
    """Create two-dimensional coordinates."""
    scaled = StandardScaler().fit_transform(features)
    if scaled.shape[1] <= 2:
        return scaled[:, :2]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(scaled)


def cluster_features(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster without labels."""
    scaled = StandardScaler().fit_transform(features)
    n_clusters = int(min(12, max(4, round(len(scaled) / 600))))
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    clusters = model.fit_predict(scaled)
    silhouette = None
    if len(set(clusters)) > 1 and len(scaled) > n_clusters:
        sample_size = min(2000, len(scaled))
        silhouette = float(
            silhouette_score(
                scaled,
                clusters,
                sample_size=sample_size,
                random_state=RANDOM_STATE,
            )
        )
    return clusters, {
        "method": "KMeans",
        "n_clusters": int(n_clusters),
        "silhouette_score": silhouette,
    }


def cluster_enrichment(labels: pd.Series, clusters: pd.Series) -> dict[str, Any]:
    """Summarize label enrichment after clustering."""
    result: dict[str, Any] = {}
    for cluster_id in sorted(clusters.unique()):
        mask = clusters.eq(cluster_id)
        result[str(cluster_id)] = {
            "row_count": int(mask.sum()),
            "label_counts": label_counts_dict(labels.loc[mask]),
            "positive_fraction_labeled": (
                float(labels.loc[mask].eq(1).sum() / labels.loc[mask].notna().sum())
                if labels.loc[mask].notna().sum()
                else None
            ),
        }
    return result


def build_outputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build cluster table and metrics."""
    data = read_csv_text(INPUT_PATH)
    labels = label_series(data)
    features, feature_source, feature_info = load_features(data)
    coords = reduce_for_plot(features)
    clusters, cluster_info = cluster_features(features)
    output = pd.DataFrame(
        {
            "row_id": np.arange(len(data), dtype=int),
            "sample_name": data["sample_name"] if "sample_name" in data.columns else "",
            "label": labels.astype("Int64"),
            "target_region_group": target_region_group_series(data),
            "paired_light_status": light_status_series(data),
            "cluster_id": clusters.astype(int),
            "landscape_component_1": coords[:, 0],
            "landscape_component_2": coords[:, 1],
        }
    )
    cluster_series = pd.Series(clusters, index=data.index)
    metrics = {
        "status": "available",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "feature_source": feature_source,
        "feature_info": feature_info,
        "cluster_info": cluster_info,
        "assigned_row_count": int(output["cluster_id"].notna().sum()),
        "assigned_row_fraction": float(output["cluster_id"].notna().mean()),
        "cluster_size_counts": value_counts_dict(output["cluster_id"]),
        "label_enrichment_after_clustering": cluster_enrichment(labels, cluster_series),
        "labels_used_for_clustering": False,
        "artifacts": {
            "clusters_csv": str(OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return output, metrics


def save_figure(table: pd.DataFrame) -> None:
    """Save landscape scatter by cluster."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        table["landscape_component_1"],
        table["landscape_component_2"],
        c=table["cluster_id"],
        cmap="tab10",
        s=10,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Unsupervised antibody sequence landscape")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown report."""
    lines = [
        "# Unsupervised Antibody Sequence Landscape",
        "",
        "Clustering used sequence-derived features only. Labels were used after",
        "clustering for enrichment summaries, not for assigning clusters.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {metrics['row_count']} |",
        f"| Feature source | {metrics['feature_source']} |",
        f"| Cluster method | {metrics['cluster_info']['method']} |",
        f"| Number of clusters | {metrics['cluster_info']['n_clusters']} |",
        f"| Assigned row fraction | {metrics['assigned_row_fraction']:.4f} |",
        f"| Silhouette score | {metrics['cluster_info']['silhouette_score']} |",
        "",
        "## Cluster Sizes",
        "",
        "| Cluster | Rows |",
        "|---|---:|",
    ]
    for key, value in metrics["cluster_size_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Label Enrichment After Clustering", ""])
    lines.extend(["| Cluster | Rows | Label 0 | Label 1 | Positive fraction among labeled |", "|---|---:|---:|---:|---:|"])
    for key, summary in metrics["label_enrichment_after_clustering"].items():
        positive_fraction = summary["positive_fraction_labeled"]
        fraction_text = "n/a" if positive_fraction is None else f"{positive_fraction:.4f}"
        lines.append(
            f"| {key} | {summary['row_count']} | {summary['label_counts']['0']} | "
            f"{summary['label_counts']['1']} | {fraction_text} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    table, metrics = build_outputs()
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV_PATH, index=False)
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    save_figure(table)
    print(
        "Unsupervised landscape complete: "
        f"rows={metrics['row_count']}, clusters={metrics['cluster_info']['n_clusters']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
