#!/usr/bin/env python3
"""Multiscale hierarchical clustering view for the 52-system distance matrix."""

from __future__ import annotations

import argparse
from .plot_utils import (
    cluster_rgb_image,
    cluster_size_profile,
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
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments/classification/results/systems52"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/classification/figures" / "dysts52_hierarchical_multiscale"
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


def linkage_from_distance(distance: np.ndarray, method: str) -> np.ndarray:
    return linkage(
        squareform(distance, checks=False), method=method, optimal_ordering=True
    )


def cut_height_for_k(z: np.ndarray, n_leaves: int, k: int) -> float:
    if k <= 1:
        return float(z[-1, 2])
    if k >= n_leaves:
        return 0.0
    lower_idx = n_leaves - k - 1
    upper_idx = n_leaves - k
    lower = 0.0 if lower_idx < 0 else float(z[lower_idx, 2])
    upper = float(z[upper_idx, 2]) if upper_idx < len(z) else float(z[-1, 2])
    if upper <= lower:
        return lower
    return 0.5 * (lower + upper)


def assignments_for_k_values(
    z: np.ndarray, order: list[int], k_values: list[int]
) -> dict[int, np.ndarray]:
    assignments: dict[int, np.ndarray] = {}
    for k in k_values:
        raw = fcluster(z, t=int(k), criterion="maxclust")
        assignments[int(k)] = remap_labels_by_leaf_order(raw, order)
    return assignments


def plot_multiscale_dendrogram(
    path_base: Path,
    names: list[str],
    distance: np.ndarray,
    z: np.ndarray,
    k_values: list[int],
    assignments: dict[int, np.ndarray],
    dataset_label: str,
    method: str,
    dpi: int,
    formats: list[str],
) -> list[Path]:
    n = len(names)
    fig_width = max(17.5, min(32.0, 0.42 * n + 8.5))
    fig_height = 10.2 if len(k_values) <= 6 else 10.9
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[10.0, 2.8],
        height_ratios=[7.0, 1.55, 0.8],
        hspace=0.05,
        wspace=0.04,
    )
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0:, 1])
    ax_strips = fig.add_subplot(gs[1, 0], sharex=ax_tree)
    ax_caption = fig.add_subplot(gs[2, 0])

    labels = [display_name(x) for x in names]
    dendro = dendrogram(
        z,
        labels=labels,
        ax=ax_tree,
        leaf_rotation=60,
        leaf_font_size=6.2 if n > 40 else 8.0,
        color_threshold=None,
        above_threshold_color="#333333",
    )
    order = [int(x) for x in dendro["leaves"]]
    max_height = float(np.max(z[:, 2]))
    ax_tree.set_title(
        f"{dataset_label} content-only distance: multiscale hierarchical clustering",
        fontsize=15,
        pad=10,
    )
    ax_tree.set_ylabel(f"{method}-linkage distance")
    ax_tree.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax_tree.tick_params(axis="x", pad=1)

    line_colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(k_values)))
    x_right = ax_tree.get_xlim()[1]
    for color, k in zip(line_colors, k_values):
        h = cut_height_for_k(z, n, k)
        ax_tree.axhline(
            h, color=color, linestyle=(0, (4, 3)), linewidth=1.15, alpha=0.78
        )
        ax_tree.annotate(
            f"k={k}, h={h:.3f}",
            xy=(x_right, h),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=color,
            clip_on=False,
        )
    ax_tree.set_ylim(0.0, max_height * 1.08)

    strip_image = cluster_rgb_image(assignments, order, k_values)
    ax_strips.imshow(strip_image, aspect="auto", extent=[0, 10 * n, len(k_values), 0])
    ax_strips.set_yticks(np.arange(len(k_values)) + 0.5)
    ax_strips.set_yticklabels([f"k={k}" for k in k_values], fontsize=9)
    ax_strips.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_strips.set_ylabel("cluster cut", fontsize=9)
    for y in range(len(k_values) + 1):
        ax_strips.axhline(y, color="white", linewidth=0.9)
    for x in range(n + 1):
        ax_strips.axvline(10 * x, color="white", linewidth=0.25, alpha=0.45)
    for spine in ax_strips.spines.values():
        spine.set_visible(False)

    ax_info.axis("off")
    ax_info.text(
        0.0, 0.99, "Cluster-size profiles", fontsize=11, weight="bold", va="top"
    )
    y = 0.94
    for k in k_values:
        sizes = cluster_size_profile(assignments[k])
        size_text = ", ".join(str(x) for x in sizes)
        h = cut_height_for_k(z, n, k)
        ax_info.text(0.0, y, f"k={k}, h={h:.3f}", fontsize=9, weight="bold", va="top")
        ax_info.text(0.0, y - 0.035, size_text, fontsize=8.2, va="top", wrap=True)
        y -= 0.095 if len(sizes) <= 8 else 0.125
    vals = distance[np.triu_indices_from(distance, 1)]
    ax_info.text(0.0, 0.16, "Distance quantiles", fontsize=11, weight="bold", va="top")
    ax_info.text(
        0.0,
        0.115,
        f"p10={np.percentile(vals, 10):.3f}\n"
        f"p25={np.percentile(vals, 25):.3f}\n"
        f"p50={np.percentile(vals, 50):.3f}\n"
        f"p75={np.percentile(vals, 75):.3f}\n"
        f"p90={np.percentile(vals, 90):.3f}",
        fontsize=8.6,
        va="top",
    )

    ax_caption.axis("off")
    ax_caption.text(
        0.0,
        0.7,
        "Color strips show cluster membership of the same dendrogram leaves under progressively finer cuts. "
        "Large singleton blocks at small k indicate outlier-dominated upper-level structure; broader color changes at larger k reveal core substructure.",
        fontsize=9,
        va="center",
    )

    saved: list[Path] = []
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.08, top=0.93)
    for fmt in formats:
        out_path = path_base.with_suffix(f".{fmt}")
        fig.savefig(
            out_path, dpi=dpi if fmt.lower() == "png" else None, bbox_inches="tight"
        )
        saved.append(out_path)
    plt.close(fig)
    return saved


def write_assignments(
    path: Path, names: list[str], assignments: dict[int, np.ndarray]
) -> None:
    k_values = sorted(assignments)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", *[f"k{k}_cluster" for k in k_values]])
        for i, name in enumerate(names):
            writer.writerow([name, *[int(assignments[k][i]) for k in k_values]])


def write_cluster_summary(
    path: Path,
    names: list[str],
    order: list[int],
    z: np.ndarray,
    assignments: dict[int, np.ndarray],
) -> None:
    n = len(names)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["k", "cut_height", "cluster", "size", "members_in_dendrogram_order"]
        )
        for k in sorted(assignments):
            labels = assignments[k]
            for cluster_id in sorted(set(int(x) for x in labels)):
                members = [
                    names[int(i)] for i in order if int(labels[int(i)]) == cluster_id
                ]
                writer.writerow(
                    [
                        k,
                        f"{cut_height_for_k(z, n, k):.12g}",
                        cluster_id,
                        len(members),
                        ";".join(members),
                    ]
                )


def write_merge_heights(path: Path, z: np.ndarray, n_leaves: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["merge_index", "height", "cluster_size_after_merge", "clusters_remaining"]
        )
        for i, row in enumerate(z, start=1):
            writer.writerow([i, f"{float(row[2]):.12g}", int(row[3]), n_leaves - i])


def parse_k_values(text: str, n: int) -> list[int]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["average", "complete", "single"],
    )
    parser.add_argument("--k-values", type=str, default="3,5,7,10,12,15")
    parser.add_argument("--dataset-label", type=str, default="DYSTS-52")
    parser.add_argument("--file-prefix", type=str, default="dysts52")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
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
    output_dir.mkdir(parents=True, exist_ok=True)

    names, matrix = read_matrix_csv(result_dir / "distance_matrix.csv")
    distance = symmetrize_distance(matrix)
    z = linkage_from_distance(distance, args.method)
    dendro = dendrogram(z, no_plot=True)
    order = [int(x) for x in dendro["leaves"]]
    k_values = parse_k_values(args.k_values, len(names))
    assignments = assignments_for_k_values(z, order, k_values)

    k_suffix = "_".join(f"k{k}" for k in k_values)
    stem = f"{args.file_prefix}_hierarchical_{args.method}_multiscale_{k_suffix}"
    formats = [
        x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()
    ]
    saved_figures = plot_multiscale_dendrogram(
        output_dir / stem,
        names,
        distance,
        z,
        k_values,
        assignments,
        args.dataset_label,
        args.method,
        args.dpi,
        formats,
    )
    assignments_path = output_dir / f"{stem}_assignments.csv"
    summary_path = output_dir / f"{stem}_cluster_summary.csv"
    heights_path = output_dir / f"{stem}_merge_heights.csv"
    write_assignments(assignments_path, names, assignments)
    write_cluster_summary(summary_path, names, order, z, assignments)
    write_merge_heights(heights_path, z, len(names))

    for path in saved_figures:
        print(f"Saved {path}")
    print(f"Saved {assignments_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {heights_path}")


if __name__ == "__main__":
    main()
