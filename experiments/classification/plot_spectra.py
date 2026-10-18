#!/usr/bin/env python3
"""Plot Lyapunov spectrum landscapes for representative small/large distances."""

from __future__ import annotations

import argparse
from .plot_utils import display_name
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments/classification/results/systems52"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/classification/figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def valid_count_by_pair(eval_rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in eval_rows:
        pair_id = row["pair_id"]
        ok = row.get("status") == "OK"
        lambdas = [
            safe_float(row.get(k, "")) for k in ("lambda1", "lambda2", "lambda3")
        ]
        if ok and all(math.isfinite(v) for v in lambdas):
            counts[pair_id] = counts.get(pair_id, 0) + 1
    return counts


def select_pairs(
    component_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    alpha_count: int,
) -> list[dict[str, str]]:
    rows = [dict(row) for row in component_rows]
    counts = valid_count_by_pair(eval_rows)
    for row in rows:
        row["distance_value"] = safe_float(row["distance"])
        row["valid_alpha_count"] = counts.get(row["pair_id"], 0)

    small = min(rows, key=lambda r: (r["distance_value"], r["pair_id"]))

    complete_rows = [
        r
        for r in rows
        if int(r["valid_alpha_count"]) == alpha_count
        and r["pair_id"] != small["pair_id"]
    ]
    if len(complete_rows) >= 2:
        large_rows = sorted(
            complete_rows, key=lambda r: (-r["distance_value"], r["pair_id"])
        )[:2]
        for row in large_rows:
            row["selection_rule"] = "large_complete_spectrum"
    else:
        large_rows = sorted(
            [r for r in rows if r["pair_id"] != small["pair_id"]],
            key=lambda r: (-r["distance_value"], r["pair_id"]),
        )[:2]
        for row in large_rows:
            row["selection_rule"] = "large_overall_fallback"

    small["selection_rule"] = "smallest_distance"
    return [small, *large_rows]


def spectra_for_pair(
    eval_rows: list[dict[str, str]], pair_id: str, alphas: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    spectra = np.full((len(alphas), 3), np.nan, dtype=float)
    status = ["missing"] * len(alphas)
    for row in eval_rows:
        if row["pair_id"] != pair_id:
            continue
        idx = int(row["alpha_index"])
        if idx < 0 or idx >= len(alphas):
            continue
        status[idx] = row.get("status", "")
        if row.get("status") == "OK":
            spectra[idx, :] = [
                safe_float(row.get(k, "")) for k in ("lambda1", "lambda2", "lambda3")
            ]
    return spectra, status


def write_selected_csv(path: Path, selected: list[dict[str, str]]) -> None:
    fieldnames = [
        "selection_rule",
        "pair_id",
        "system_a",
        "system_b",
        "distance",
        "endpoint_gap",
        "barrier",
        "roughness",
        "chaos_loss",
        "valid_alpha_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pair_label(row: dict[str, str]) -> str:
    system_a = display_name(row["system_a"])
    system_b = display_name(row["system_b"])
    return f"{system_a}-{system_b}"


def plot_landscape(
    selected: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    alphas: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    spectra_items = []
    finite_values: list[float] = []
    for row in selected:
        spectra, status = spectra_for_pair(eval_rows, row["pair_id"], alphas)
        spectra_items.append((row, spectra, status))
        finite_values.extend(
            [float(v) for v in spectra.reshape(-1) if math.isfinite(float(v))]
        )

    if finite_values:
        lo, hi = np.percentile(np.asarray(finite_values, dtype=float), [3, 97])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = min(finite_values), max(finite_values)
    else:
        lo, hi = -1.0, 1.0

    fig = plt.figure(figsize=(13.6, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=len(spectra_items),
        ncols=3,
        width_ratios=[1.45, 1.95, 0.07],
        hspace=0.56,
        wspace=0.26,
    )

    line_colors = ["#1b4965", "#d95f02", "#3a7d44"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    row_label_fontsize = 18.5
    axis_label_fontsize = 15.0
    tick_fontsize = 13.0
    heat_tick_fontsize = 15.0
    legend_fontsize = 12.0

    for row_index, (row, spectra, status) in enumerate(spectra_items):
        ax_line = fig.add_subplot(gs[row_index, 0])
        ax_heat = fig.add_subplot(gs[row_index, 1])
        ax_cbar = fig.add_subplot(gs[row_index, 2])
        line_box = ax_line.get_position()
        heat_box = ax_heat.get_position()
        fig.text(
            0.5 * (line_box.x0 + heat_box.x1),
            max(line_box.y1, heat_box.y1) + 0.018,
            pair_label(row),
            ha="center",
            va="bottom",
            fontsize=row_label_fontsize,
            color="#202020",
        )

        for dim in range(3):
            ax_line.plot(
                alphas,
                spectra[:, dim],
                marker="o",
                linewidth=1.9,
                markersize=4.2,
                color=line_colors[dim],
                label=labels[dim],
            )
        ax_line.axhline(0.0, color="#202020", linewidth=0.8, alpha=0.55)
        bad = np.array(
            [
                (status[i] != "OK")
                or (not math.isfinite(float(spectra[i, 0])))
                or (float(spectra[i, 0]) <= 0.0)
                for i in range(len(alphas))
            ],
            dtype=bool,
        )
        y_min = np.nanmin(spectra) if np.any(np.isfinite(spectra)) else -1.0
        y_max = np.nanmax(spectra) if np.any(np.isfinite(spectra)) else 1.0
        y_pad = 0.08 * max(1.0, y_max - y_min)
        ax_line.set_ylim(y_min - y_pad, y_max + y_pad)
        for alpha, is_bad in zip(alphas, bad):
            if is_bad:
                ax_line.axvline(
                    float(alpha), color="#b00020", linewidth=0.8, alpha=0.14
                )
        failed = [i for i, s in enumerate(status) if s != "OK"]
        if failed:
            ax_line.scatter(
                alphas[failed],
                np.full(len(failed), y_min - 0.03 * max(1.0, y_max - y_min)),
                marker="x",
                color="#b00020",
                s=34,
                label="failed",
            )

        ax_line.set_ylabel(
            "Lyapunov exponent", fontsize=axis_label_fontsize, labelpad=8
        )
        ax_line.set_xlim(float(alphas[0]), float(alphas[-1]))
        ax_line.set_xticks(alphas)
        ax_line.tick_params(axis="x", labelrotation=35, labelsize=tick_fontsize)
        ax_line.tick_params(axis="y", labelsize=tick_fontsize)
        ax_line.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.8)
        if row_index == len(spectra_items) - 1:
            ax_line.set_xlabel(
                r"fusion weight $\alpha$", fontsize=axis_label_fontsize, labelpad=7
            )
        if row_index == 0:
            ax_line.legend(loc="best", fontsize=legend_fontsize, frameon=True)

        masked = np.ma.masked_invalid(spectra.T)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#d9d9d9")
        image = ax_heat.imshow(
            masked, aspect="equal", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest"
        )
        ax_heat.set_yticks([0, 1, 2], labels=labels)
        ax_heat.set_xticks(np.arange(len(alphas)), labels=[f"{a:.3g}" for a in alphas])
        ax_heat.tick_params(axis="x", labelrotation=35, labelsize=tick_fontsize)
        ax_heat.tick_params(axis="y", labelsize=heat_tick_fontsize)
        ax_heat.set_box_aspect(3.0 / len(alphas))
        if row_index == len(spectra_items) - 1:
            ax_heat.set_xlabel(
                r"fusion weight $\alpha$", fontsize=axis_label_fontsize, labelpad=7
            )
        for failed_idx in failed:
            ax_heat.scatter(
                failed_idx, 1, marker="x", color="#111111", s=36, linewidths=1.2
            )
        cbar = fig.colorbar(image, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    component_rows = read_csv(args.result_dir / "pair_components.csv")
    eval_rows = read_csv(args.result_dir / "eval_records.csv")
    alphas = np.array(
        sorted(
            {
                safe_float(row["alpha"])
                for row in eval_rows
                if math.isfinite(safe_float(row["alpha"]))
            }
        )
    )

    selected = select_pairs(component_rows, eval_rows, alpha_count=len(alphas))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_csv(
        args.output_dir / "lyapunov_spectrum_landscape_selected_pairs.csv", selected
    )
    plot_landscape(
        selected,
        eval_rows,
        alphas,
        args.output_dir / "lyapunov_spectrum_landscape_distance_examples.png",
        args.dpi,
    )

    print("Selected pairs:")
    for row in selected:
        print(
            f"{row['selection_rule']}: {row['pair_id']} "
            f"distance={row['distance']} valid_alpha_count={row.get('valid_alpha_count', '')}"
        )
    print(
        f"Saved: {args.output_dir / 'lyapunov_spectrum_landscape_distance_examples.png'}"
    )
    print(
        f"Saved: {args.output_dir / 'lyapunov_spectrum_landscape_selected_pairs.csv'}"
    )


if __name__ == "__main__":
    main()
