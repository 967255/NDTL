#!/usr/bin/env python3
"""Strips-only multiscale hierarchical clustering view with a spectral reference row."""

from __future__ import annotations

import argparse
from .plot_utils import (
    cluster_rgb_image,
    cluster_size_profile_text as cluster_size_profile,
    display_name,
    remap_labels_by_leaf_order,
    symmetrize_distance,
    read_matrix_csv,
)
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments/classification/results/systems52"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/classification/figures" / "dysts52_hierarchical_multiscale"
)
DEFAULT_SPECTRAL_ASSIGNMENTS = (
    DEFAULT_RESULT_DIR
    / "spectral_embedding_content"
    / "spectral_embedding_k7_assignments.csv"
)

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
    "#edc948",
    "#af7aa1",
    "#76b7b2",
    "#e15759",
]


def read_spectral_assignments(path: Path, names: list[str]) -> np.ndarray:
    labels_by_name: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if (
            reader.fieldnames is None
            or "system" not in reader.fieldnames
            or "cluster" not in reader.fieldnames
        ):
            raise ValueError(f"Unexpected spectral assignment header: {path}")
        for row in reader:
            labels_by_name[row["system"]] = int(row["cluster"])
    missing = [name for name in names if name not in labels_by_name]
    if missing:
        raise ValueError(f"Missing spectral assignments for: {', '.join(missing)}")
    return np.array([labels_by_name[name] for name in names], dtype=int)


def parse_k_range(text: str, n: int) -> list[int]:
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        raw = list(range(start, end + 1))
    else:
        raw = [int(x.strip()) for x in text.split(",") if x.strip()]
    k_values: list[int] = []
    for k in raw:
        if k < 2 or k > n:
            raise ValueError(f"k must be in [2, {n}], got {k}")
        if k not in k_values:
            k_values.append(k)
    if not k_values:
        raise ValueError("At least one k value is required")
    return k_values


def match_target_colors_to_reference(
    target_labels: np.ndarray, reference_labels: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Map target clusters onto reference cluster color ids by maximum overlap."""
    target_values = sorted(int(x) for x in np.unique(target_labels))
    reference_values = sorted(int(x) for x in np.unique(reference_labels))
    overlap = np.zeros((len(target_values), len(reference_values)), dtype=float)
    for i, target in enumerate(target_values):
        target_mask = target_labels == target
        for j, reference in enumerate(reference_values):
            overlap[i, j] = float(np.sum(target_mask & (reference_labels == reference)))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {
        target_values[int(i)]: reference_values[int(j)]
        for i, j in zip(row_ind, col_ind)
    }
    for i, target in enumerate(target_values):
        if target not in mapping:
            mapping[target] = reference_values[int(np.argmax(overlap[i]))]

    matched = np.array([mapping[int(label)] for label in target_labels], dtype=int)
    return matched, mapping


def write_ordered_assignments(
    path: Path,
    names: list[str],
    order: list[int],
    hierarchical_assignments: dict[int, np.ndarray],
    spectral_raw: np.ndarray,
    spectral_display: np.ndarray,
    spectral_color_labels: np.ndarray,
    spectral_color_reference_k: int,
) -> None:
    k_values = sorted(hierarchical_assignments)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order",
                "system",
                *[f"hierarchical_k{k}" for k in k_values],
                "spectral_k7_raw",
                "spectral_k7_display",
                f"spectral_k7_color_matched_to_hierarchical_k{spectral_color_reference_k}",
            ]
        )
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    names[int(idx)],
                    *[int(hierarchical_assignments[k][int(idx)]) for k in k_values],
                    int(spectral_raw[int(idx)]),
                    int(spectral_display[int(idx)]),
                    int(spectral_color_labels[int(idx)]),
                ]
            )


def write_cluster_sizes(
    path: Path,
    hierarchical_assignments: dict[int, np.ndarray],
    spectral_raw: np.ndarray,
    spectral_display: np.ndarray,
    spectral_color_labels: np.ndarray,
    spectral_color_reference_k: int,
    spectral_color_mapping: dict[int, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "cluster_size_profile_descending"])
        for k in sorted(hierarchical_assignments):
            writer.writerow(
                [
                    f"hierarchical_k{k}",
                    cluster_size_profile(hierarchical_assignments[k]),
                ]
            )
        writer.writerow(["spectral_k7_raw", cluster_size_profile(spectral_raw)])
        writer.writerow(["spectral_k7_display", cluster_size_profile(spectral_display)])
        writer.writerow(
            [
                f"spectral_k7_color_matched_to_hierarchical_k{spectral_color_reference_k}",
                cluster_size_profile(spectral_color_labels),
            ]
        )
        mapping_text = ";".join(
            f"spectral_{k}->hierarchical_{v}"
            for k, v in sorted(spectral_color_mapping.items())
        )
        writer.writerow(
            [
                f"spectral_color_mapping_to_hierarchical_k{spectral_color_reference_k}",
                mapping_text,
            ]
        )


def plot_strips(
    path_base: Path,
    names: list[str],
    order: list[int],
    row_labels: list[str],
    rows: list[np.ndarray],
    dataset_label: str,
    method: str,
    dpi: int,
    formats: list[str],
    paper_figure: bool,
) -> list[Path]:
    n = len(names)
    n_rows = len(rows)
    ordered_names = [display_name(names[int(idx)]) for idx in order]
    image = cluster_rgb_image(rows, order)

    fig_width = max(32.0, min(52.0, 0.86 * n + 8.0))
    fig_height = max(16.5, fig_width * n_rows / n + 9.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    ax.imshow(image, aspect="equal", extent=[0, n, n_rows, 0])

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(ordered_names, rotation=68, ha="right", va="top", fontsize=18)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.xaxis.tick_bottom()

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=18)
    ax.tick_params(axis="y", length=0, pad=11)
    if not paper_figure:
        ax.set_ylabel("cluster assignment", fontsize=18)

    ax.set_xticks(np.arange(n + 1), minor=True)
    ax.set_yticks(np.arange(n_rows + 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(0, n)
    ax.set_ylim(n_rows, 0)

    spectral_boundary = n_rows - 1
    ax.axhline(spectral_boundary, color="#222222", linewidth=1.35)

    if not paper_figure:
        ax.set_title(
            f"{dataset_label}: hierarchical {method}-linkage cuts k=2..15 with spectral k=7 reference",
            fontsize=20,
            pad=16,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#777777")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.56)
    saved: list[Path] = []
    for fmt in formats:
        out_path = path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        saved.append(out_path)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--spectral-assignments", type=Path, default=DEFAULT_SPECTRAL_ASSIGNMENTS
    )
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["average", "complete", "single"],
    )
    parser.add_argument("--k-range", type=str, default="2-15")
    parser.add_argument("--spectral-color-reference-k", type=int, default=None)
    parser.add_argument(
        "--dataset-label", type=str, default="DYSTS-52 content-only distance"
    )
    parser.add_argument(
        "--file-prefix", type=str, default="dysts52_hierarchical_strips"
    )
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    parser.add_argument("--paper-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = (
        args.result_dir
        if args.result_dir.is_absolute()
        else PROJECT_ROOT / args.result_dir
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    spectral_path = (
        args.spectral_assignments
        if args.spectral_assignments.is_absolute()
        else PROJECT_ROOT / args.spectral_assignments
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    names, matrix = read_matrix_csv(result_dir / "distance_matrix.csv")
    distance = symmetrize_distance(matrix, reject_negative=False)
    z = linkage(
        squareform(distance, checks=False), method=args.method, optimal_ordering=True
    )
    order = [int(x) for x in dendrogram(z, no_plot=True)["leaves"]]
    k_values = parse_k_range(args.k_range, len(names))

    hierarchical_assignments: dict[int, np.ndarray] = {}
    for k in k_values:
        raw = fcluster(z, t=int(k), criterion="maxclust")
        hierarchical_assignments[k] = remap_labels_by_leaf_order(raw, order)

    spectral_raw = read_spectral_assignments(spectral_path, names)
    spectral_display = remap_labels_by_leaf_order(spectral_raw, order)
    spectral_color_reference_k = int(args.spectral_color_reference_k or max(k_values))
    if spectral_color_reference_k not in hierarchical_assignments:
        raise ValueError(
            f"--spectral-color-reference-k={spectral_color_reference_k} is not included in --k-range"
        )
    spectral_color_labels, spectral_color_mapping = match_target_colors_to_reference(
        spectral_raw,
        hierarchical_assignments[spectral_color_reference_k],
    )

    row_labels = [f"k={k}" for k in k_values] + ["Spec. k=7"]
    rows = [hierarchical_assignments[k] for k in k_values] + [spectral_color_labels]
    stem = f"{args.file_prefix}_{args.method}_k{min(k_values)}_to_k{max(k_values)}_plus_spectral_k7"
    formats = [
        x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()
    ]

    saved_figures = plot_strips(
        output_dir / stem,
        names,
        order,
        row_labels,
        rows,
        args.dataset_label,
        args.method,
        args.dpi,
        formats,
        args.paper_figure,
    )
    assignments_path = output_dir / f"{stem}_ordered_assignments.csv"
    sizes_path = output_dir / f"{stem}_cluster_sizes.csv"
    write_ordered_assignments(
        assignments_path,
        names,
        order,
        hierarchical_assignments,
        spectral_raw,
        spectral_display,
        spectral_color_labels,
        spectral_color_reference_k,
    )
    write_cluster_sizes(
        sizes_path,
        hierarchical_assignments,
        spectral_raw,
        spectral_display,
        spectral_color_labels,
        spectral_color_reference_k,
        spectral_color_mapping,
    )

    for path in saved_figures:
        print(f"Saved {path}")
    print(f"Saved {assignments_path}")
    print(f"Saved {sizes_path}")


if __name__ == "__main__":
    main()
