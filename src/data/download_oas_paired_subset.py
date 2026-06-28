"""Download a small paired OAS subset for background retrieval.

The script stores compressed CSV files under data/raw/oas/. It prints only
aggregate download status and file paths, not file contents.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "oas"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "oas_download_status.json"
TIMEOUT_SECONDS = 90
CHUNK_SIZE = 1024 * 1024

OAS_URLS = [
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Alsoiussi_2020/csv/SRR11528761_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Alsoiussi_2020/csv/SRR11528762_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Eccles_2020/csv/SRR10358523_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Eccles_2020/csv/SRR10358524_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Eccles_2020/csv/SRR10358525_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Goldstein_2019/csv/SRR9179273_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Goldstein_2019/csv/SRR9179274_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Goldstein_2019/csv/SRR9179275_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Goldstein_2019/csv/SRR9179276_paired.csv.gz",
    "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Goldstein_2019/csv/SRR9179277_paired.csv.gz",
]


def filename_from_url(url: str) -> str:
    """Return the final URL path component."""
    return url.rstrip("/").split("/")[-1]


def download_one(url: str, output_path: Path) -> dict[str, Any]:
    """Download one file unless it already exists."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "file_path": str(output_path.relative_to(PROJECT_ROOT)),
            "status": "exists",
            "bytes": int(output_path.stat().st_size),
        }

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "antibody-prioritization-oas-background/1.0"},
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp_path.replace(output_path)
        return {
            "file_path": str(output_path.relative_to(PROJECT_ROOT)),
            "status": "downloaded",
            "bytes": int(output_path.stat().st_size),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "file_path": str(output_path.relative_to(PROJECT_ROOT)),
            "status": "failed",
            "error_type": type(exc).__name__,
            "bytes": 0,
        }


def write_metrics(payload: dict[str, Any]) -> None:
    """Write aggregate download metrics."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "metrics").mkdir(parents=True, exist_ok=True)

    records = []
    for url in OAS_URLS:
        records.append(download_one(url, RAW_DIR / filename_from_url(url)))

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
    payload = {
        "status": "available" if any(r["status"] in {"downloaded", "exists"} for r in records) else "failed",
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "file_count": len(records),
        "status_counts": status_counts,
        "files": records,
    }
    write_metrics(payload)
    print(
        "download_status="
        f"{payload['status']}; files={payload['file_count']}; "
        f"status_counts={payload['status_counts']}; raw_dir={payload['raw_dir']}",
        flush=True,
    )
    return 0 if payload["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
