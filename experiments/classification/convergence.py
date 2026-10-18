from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ndtl.paths import ROOT as PROJECT_ROOT, inside_project as resolve_inside_project

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments/classification/results/systems20"


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cae_rows(result_dir: Path) -> list[dict[str, str]]:
    final_path = result_dir / "cae_records.csv"
    if final_path.exists():
        return read_rows(final_path)

    rows_by_pair: dict[str, dict[str, str]] = {}
    for path in sorted((result_dir / "shards").glob("shard_*_of_*/cae_records.csv")):
        for row in read_rows(path):
            rows_by_pair[row["pair_id"]] = row
    return sorted(rows_by_pair.values(), key=lambda r: to_int(r.get("pair_index")))


def quantile(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = q * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    weight = pos - lo
    return finite[lo] * (1.0 - weight) + finite[hi] * weight


def summarize(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            k: float("nan")
            for k in ["min", "p25", "median", "p75", "p90", "max", "mean"]
        }
    return {
        "min": min(finite),
        "p25": quantile(finite, 0.25),
        "median": quantile(finite, 0.50),
        "p75": quantile(finite, 0.75),
        "p90": quantile(finite, 0.90),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def write_outliers(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_index",
        "pair_id",
        "status",
        "train_batches",
        "best_loss",
        "final_loss",
        "final_lr",
        "recon_mse_a",
        "recon_mse_b",
        "recon_mse_max",
        "issue",
        "model_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = format_float(value)
                out[key] = value
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check CAE reconstruction convergence for content-only pair CAEs."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--expected-pairs", type=int, default=190)
    parser.add_argument("--warn-recon-mse", type=float, default=5e-3)
    parser.add_argument("--fail-recon-mse", type=float, default=2e-2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if incomplete, failed, or above fail threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = resolve_inside_project(args.result_dir)
    rows = load_cae_rows(result_dir)

    parsed: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for row in rows:
        recon_a = to_float(row.get("recon_mse_a"))
        recon_b = to_float(row.get("recon_mse_b"))
        recon_max = max(recon_a, recon_b)
        item = {
            "pair_index": to_int(row.get("pair_index")),
            "pair_id": row.get("pair_id", ""),
            "status": row.get("status", ""),
            "train_batches": to_int(row.get("train_batches")),
            "best_loss": to_float(row.get("best_loss")),
            "final_loss": to_float(row.get("final_loss")),
            "final_lr": to_float(row.get("final_lr")),
            "recon_mse_a": recon_a,
            "recon_mse_b": recon_b,
            "recon_mse_max": recon_max,
            "model_path": row.get("model_path", ""),
        }
        parsed.append(item)

        issues = []
        if item["status"] != "OK":
            issues.append("status_not_ok")
        if not math.isfinite(recon_max):
            issues.append("missing_recon_mse")
        elif recon_max > float(args.fail_recon_mse):
            issues.append("above_fail_recon_mse")
        elif recon_max > float(args.warn_recon_mse):
            issues.append("above_warn_recon_mse")
        if issues:
            out = dict(item)
            out["issue"] = ";".join(issues)
            outliers.append(out)

    recon_max_values = [row["recon_mse_max"] for row in parsed if row["status"] == "OK"]
    best_losses = [row["best_loss"] for row in parsed if row["status"] == "OK"]
    final_losses = [row["final_loss"] for row in parsed if row["status"] == "OK"]

    summary = {
        "result_dir": str(result_dir),
        "expected_pairs": int(args.expected_pairs),
        "num_records": len(parsed),
        "num_ok": sum(1 for row in parsed if row["status"] == "OK"),
        "num_failed": sum(1 for row in parsed if row["status"] != "OK"),
        "num_missing": max(0, int(args.expected_pairs) - len(parsed)),
        "warn_recon_mse": float(args.warn_recon_mse),
        "fail_recon_mse": float(args.fail_recon_mse),
        "recon_mse_max_summary": summarize(recon_max_values),
        "best_loss_summary": summarize(best_losses),
        "final_loss_summary": summarize(final_losses),
        "num_outliers": len(outliers),
        "outliers": outliers[:20],
    }

    report_path = result_dir / "cae_convergence_report.json"
    outlier_path = result_dir / "cae_reconstruction_outliers.csv"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_outliers(outlier_path, outliers)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict:
        hard_fail = (
            summary["num_missing"] > 0
            or summary["num_failed"] > 0
            or any(
                "above_fail_recon_mse" in str(row.get("issue", ""))
                or "missing_recon_mse" in str(row.get("issue", ""))
                for row in outliers
            )
        )
        if hard_fail:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
