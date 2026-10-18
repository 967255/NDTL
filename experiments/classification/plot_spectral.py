#!/usr/bin/env python3
"""Spectral clustering visualizations for the 20-system content-only distance matrix."""

from __future__ import annotations

import argparse
from .plot_utils import (
    display_name,
    distance_to_affinity,
    spectral_coordinates,
    read_symmetric_matrix_csv as read_matrix_csv,
)
import csv
import os
from pathlib import Path

from ndtl.paths import ROOT as _ROOT

LOCAL_CACHE_ROOT = _ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments/classification/results/systems20"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/classification/figures" / "systems20_spectral_clustering"
)
PALETTE = [
    "#1f77b4",
    "#d95f02",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
]


def remap_labels_by_centroid(labels: np.ndarray, coords: np.ndarray) -> np.ndarray:
    stats = []
    for old in sorted(set(int(x) for x in labels)):
        pts = coords[labels == old]
        center = pts.mean(axis=0)
        stats.append((float(center[0]), float(center[1]), old))
    stats.sort()
    mapping = {old: new for new, (_, _, old) in enumerate(stats, start=1)}
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def fit_spectral_labels(
    affinity: np.ndarray, coords: np.ndarray, k: int, random_state: int
) -> np.ndarray:
    labels = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=50,
    ).fit_predict(affinity)
    return remap_labels_by_centroid(np.asarray(labels, dtype=int), coords)


def silhouette_from_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for i in range(distance.shape[0]):
        own = labels == labels[i]
        own[i] = False
        a = float(np.mean(distance[i, own])) if np.any(own) else 0.0
        b = float("inf")
        for label in sorted(set(int(x) for x in labels)):
            if label == int(labels[i]):
                continue
            mask = labels == label
            if np.any(mask):
                b = min(b, float(np.mean(distance[i, mask])))
        if np.isfinite(b):
            denom = max(a, b)
            scores.append((b - a) / denom if denom > 0 else 0.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def cluster_order(
    names: list[str], coords: np.ndarray, labels: np.ndarray
) -> list[int]:
    ordered: list[int] = []
    for label in sorted(set(int(x) for x in labels)):
        members = [i for i in range(len(names)) if int(labels[i]) == label]
        members.sort(key=lambda i: (float(coords[i, 0]), float(coords[i, 1]), names[i]))
        ordered.extend(members)
    return ordered


def add_cluster_ellipse(ax: plt.Axes, pts: np.ndarray, color: str) -> None:
    center = pts.mean(axis=0)
    if pts.shape[0] <= 1:
        ellipse = Ellipse(center, width=0.18, height=0.18, angle=0.0)
    else:
        cov = np.cov(pts.T) + 1e-5 * np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-5)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        ellipse = Ellipse(
            center,
            width=3.0 * float(np.sqrt(vals[0])),
            height=3.0 * float(np.sqrt(vals[1])),
            angle=angle,
        )
    ellipse.set_facecolor(color)
    ellipse.set_edgecolor(color)
    ellipse.set_alpha(0.15)
    ellipse.set_linewidth(1.6)
    ax.add_patch(ellipse)


def plot_embedding(
    path: Path,
    names: list[str],
    coords: np.ndarray,
    labels: np.ndarray,
    main_k: int,
    sigma: float,
    silhouette: float,
    dataset_label: str,
    paper_embedding: bool,
    dpi: int,
) -> None:
    fig_size = max(11.2, min(16.0, 0.10 * len(names) + 9.5))
    if paper_embedding:
        fig_size = max(fig_size, 12.8)
        label_font = 17.0 if len(names) <= 25 else 7.0
        point_size = 250 if len(names) <= 25 else 70
        tick_font = 18.0
        axis_font = 23.0
    else:
        label_font = 8.4 if len(names) <= 25 else 5.8
        point_size = 78 if len(names) <= 25 else 48
        tick_font = 10.0
        axis_font = 10.0
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1.0), constrained_layout=True)
    unique_labels = sorted(set(int(x) for x in labels))
    for label in unique_labels:
        color = PALETTE[(label - 1) % len(PALETTE)]
        add_cluster_ellipse(ax, coords[labels == label], color)
    for label in unique_labels:
        color = PALETTE[(label - 1) % len(PALETTE)]
        pts = coords[labels == label]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            label=f"cluster {label}",
            zorder=3,
        )

    offsets = [(6, 5), (6, -11), (-6, 5), (-6, -11), (9, 0), (-9, 0)]
    for i, name in enumerate(names):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            display_name(name),
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=label_font,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            color="#111111",
            zorder=4,
        )
        if not paper_embedding:
            ax.text(
                coords[i, 0],
                coords[i, 1],
                str(int(labels[i])),
                ha="center",
                va="center",
                fontsize=max(4.8, label_font - 1.9),
                color="white",
                zorder=5,
            )

    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    xlabel = (
        "Spectral embedding coordinate 1" if paper_embedding else "spectral embedding 1"
    )
    ylabel = (
        "Spectral embedding coordinate 2" if paper_embedding else "spectral embedding 2"
    )
    ax.set_xlabel(xlabel, fontsize=axis_font, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=axis_font, labelpad=9)
    ax.tick_params(axis="x", labelsize=tick_font)
    ax.tick_params(axis="y", labelsize=tick_font)
    if not paper_embedding:
        ax.set_title(
            f"{dataset_label} Spectral Clustering on Content-Only Distance, k={main_k}",
            fontsize=14.5,
            pad=10,
        )
        ax.text(
            0.01,
            0.01,
            f"RBF affinity; sigma={sigma:.3f}; silhouette={silhouette:.3f}",
            transform=ax.transAxes,
            fontsize=8.3,
            color="#555555",
            ha="left",
            va="bottom",
        )
        ax.legend(loc="best", frameon=True, fontsize=8.2)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.18)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_clustermap(
    path: Path,
    names: list[str],
    distance: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    main_k: int,
    dataset_label: str,
    paper_clustermap: bool,
    dpi: int,
) -> None:
    order = cluster_order(names, coords, labels)
    ordered_matrix = distance[np.ix_(order, order)]
    ordered_labels = np.array([int(labels[i]) for i in order], dtype=int)
    tick_labels = [display_name(names[i]) for i in order]
    cluster_colors = [PALETTE[(label - 1) % len(PALETTE)] for label in ordered_labels]

    if paper_clustermap:
        fig_size = max(20.0, min(23.0, 0.36 * len(order) + 3.6))
        tick_font = 8.4 if len(order) > 25 else 10.0
        fig = plt.figure(figsize=(fig_size, fig_size), constrained_layout=False)
        heat_left = 0.095
        heat_bottom = 0.175
        heat_size = 0.745
        strip_thickness = 0.026
        cbar_gap = 0.072
        cbar_width = 0.007
        cbar_height = heat_size * 0.88
        cbar_bottom = heat_bottom + 0.5 * (heat_size - cbar_height)
        ax_heat = fig.add_axes([heat_left, heat_bottom, heat_size, heat_size])
        ax_col = fig.add_axes(
            [heat_left, heat_bottom + heat_size, heat_size, strip_thickness]
        )
        ax_row = fig.add_axes(
            [heat_left - strip_thickness, heat_bottom, strip_thickness, heat_size]
        )
        ax_cbar = fig.add_axes(
            [heat_left + heat_size + cbar_gap, cbar_bottom, cbar_width, cbar_height]
        )
    else:
        fig_size = max(12.8, min(20.0, 0.28 * len(order) + 4.0))
        tick_font = 8 if len(order) <= 25 else 5.3
        fig = plt.figure(figsize=(fig_size, fig_size - 1.4), constrained_layout=False)
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[0.25, 8.5],
            height_ratios=[0.25, 8.5],
            wspace=0.02,
            hspace=0.02,
        )
        ax_col = fig.add_subplot(gs[0, 1])
        ax_row = fig.add_subplot(gs[1, 0])
        ax_heat = fig.add_subplot(gs[1, 1])
        ax_cbar = None

    color_rgb = np.array(
        [matplotlib.colors.to_rgb(c) for c in cluster_colors], dtype=float
    )
    ax_col.imshow(color_rgb.reshape(1, len(order), 3), aspect="auto")
    ax_row.imshow(color_rgb.reshape(len(order), 1, 3), aspect="auto")
    for ax in [ax_col, ax_row]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    im = ax_heat.imshow(ordered_matrix, cmap="magma_r", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(order)))
    ax_heat.set_yticks(np.arange(len(order)))
    ax_heat.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=tick_font)
    ax_heat.set_yticklabels(tick_labels, fontsize=tick_font)
    ax_heat.yaxis.tick_right()
    ax_heat.set_xlim(-0.5, len(order) - 0.5)
    ax_heat.set_ylim(len(order) - 0.5, -0.5)
    ax_heat.set_aspect("equal")
    ax_heat.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    if paper_clustermap:
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=8.0)
    else:
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.038, pad=0.12)
        cbar.set_label("content-only distance")

        legend_handles = []
        for label in sorted(set(int(x) for x in labels)):
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="none",
                    markerfacecolor=PALETTE[(label - 1) % len(PALETTE)],
                    markersize=8,
                    label=f"cluster {label}",
                )
            )
        ax_heat.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.18, 1.0),
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            f"{dataset_label} Spectral Clustering Reordered Distance Matrix, k={main_k}",
            fontsize=14.5,
            y=0.985,
        )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_k_scan(
    path: Path,
    scan_rows: list[dict[str, object]],
    main_k: int,
    dataset_label: str,
    dpi: int,
) -> None:
    ks = [int(row["k"]) for row in scan_rows]
    silhouettes = [float(row["silhouette"]) for row in scan_rows]
    size_text = [" ".join(str(x) for x in row["cluster_sizes"]) for row in scan_rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    ax.plot(ks, silhouettes, marker="o", color="#1f77b4", linewidth=2.0)
    ax.axvline(
        main_k, color="#d95f02", linestyle="--", linewidth=1.2, label=f"main k={main_k}"
    )
    for k, sil, txt in zip(ks, silhouettes, size_text):
        ax.annotate(
            txt,
            xy=(k, sil),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            color="#333333",
        )
    ax.set_title(f"{dataset_label} Spectral Clustering k Scan")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("mean silhouette on original distance")
    ax.set_xticks(ks)
    ymin = min(silhouettes)
    ymax = max(silhouettes)
    ax.set_ylim(ymin - 0.02, ymax + 0.045)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_assignments(
    path: Path, names: list[str], coords: np.ndarray, assignments: dict[int, np.ndarray]
) -> None:
    k_values = sorted(assignments)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "embedding_1",
                "embedding_2",
                *[f"k{k}_cluster" for k in k_values],
            ]
        )
        for i, name in enumerate(names):
            writer.writerow(
                [
                    name,
                    f"{coords[i, 0]:.12g}",
                    f"{coords[i, 1]:.12g}",
                    *[int(assignments[k][i]) for k in k_values],
                ]
            )


def write_summary(
    path: Path,
    names: list[str],
    coords: np.ndarray,
    distance: np.ndarray,
    assignments: dict[int, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "cluster", "size", "members", "mean_silhouette"])
        for k in sorted(assignments):
            labels = assignments[k]
            sil = silhouette_from_distance(distance.copy(), labels.copy())
            order = cluster_order(names, coords, labels)
            for cluster_id in sorted(set(int(x) for x in labels)):
                members = [names[i] for i in order if int(labels[i]) == cluster_id]
                writer.writerow([k, cluster_id, len(members), ";".join(members), sil])


def write_k_scan(path: Path, scan_rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "mean_silhouette", "cluster_sizes"])
        for row in scan_rows:
            writer.writerow(
                [
                    row["k"],
                    row["silhouette"],
                    ";".join(str(x) for x in row["cluster_sizes"]),
                ]
            )


def write_order(
    path: Path, names: list[str], coords: np.ndarray, labels: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "system", "cluster"])
        for rank, idx in enumerate(cluster_order(names, coords, labels), start=1):
            writer.writerow([rank, names[idx], int(labels[idx])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=str, default="3,4,5,6")
    parser.add_argument("--scan-k-values", type=str, default="2,3,4,5,6,7,8")
    parser.add_argument("--main-k", type=int, default=5)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=20260605)
    parser.add_argument("--dataset-label", type=str, default="Pilot-20")
    parser.add_argument("--file-prefix", type=str, default="systems20")
    parser.add_argument("--paper-embedding", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--paper-clustermap", action="store_true")
    parser.add_argument("--clustermap-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
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

    names, distance = read_matrix_csv(result_dir / "distance_matrix.csv")
    affinity, sigma = distance_to_affinity(distance, args.sigma_scale)
    coords = spectral_coordinates(affinity, args.random_state)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    scan_k_values = [int(x.strip()) for x in args.scan_k_values.split(",") if x.strip()]
    all_k = sorted(set(k_values + scan_k_values + [int(args.main_k)]))
    assignments = {
        k: fit_spectral_labels(affinity, coords, k, args.random_state) for k in all_k
    }
    main_labels = assignments[int(args.main_k)]
    main_silhouette = silhouette_from_distance(distance.copy(), main_labels.copy())

    scan_rows = []
    for k in scan_k_values:
        labels = assignments[k]
        sizes = [
            int(np.sum(labels == cluster_id))
            for cluster_id in sorted(set(int(x) for x in labels))
        ]
        scan_rows.append(
            {
                "k": k,
                "silhouette": silhouette_from_distance(distance.copy(), labels.copy()),
                "cluster_sizes": sizes,
            }
        )

    stem = f"{args.file_prefix}_spectral_k{args.main_k}"
    k_suffix = "_".join(f"k{k}" for k in k_values)
    assignment_path = (
        output_dir / f"{args.file_prefix}_spectral_assignments_{k_suffix}.csv"
    )
    if not args.clustermap_only:
        plot_embedding(
            output_dir / f"{stem}_embedding.png",
            names,
            coords,
            main_labels,
            int(args.main_k),
            sigma,
            main_silhouette,
            args.dataset_label,
            args.paper_embedding,
            args.dpi,
        )
    if not args.embedding_only:
        plot_clustermap(
            output_dir / f"{stem}_clustermap.png",
            names,
            distance,
            coords,
            main_labels,
            int(args.main_k),
            args.dataset_label,
            args.paper_clustermap,
            args.dpi,
        )
    if not args.embedding_only and not args.clustermap_only:
        plot_k_scan(
            output_dir / f"{args.file_prefix}_spectral_k_scan.png",
            scan_rows,
            int(args.main_k),
            args.dataset_label,
            args.dpi,
        )
    write_assignments(
        assignment_path, names, coords, {k: assignments[k] for k in k_values}
    )
    if not args.embedding_only and not args.clustermap_only:
        write_summary(
            output_dir / f"{args.file_prefix}_spectral_cluster_summary.csv",
            names,
            coords,
            distance,
            {k: assignments[k] for k in k_values},
        )
        write_k_scan(output_dir / f"{args.file_prefix}_spectral_k_scan.csv", scan_rows)
    write_order(output_dir / f"{stem}_cluster_order.csv", names, coords, main_labels)

    if not args.clustermap_only:
        print(f"Saved {output_dir / f'{stem}_embedding.png'}")
    if not args.embedding_only:
        print(f"Saved {output_dir / f'{stem}_clustermap.png'}")
    if not args.embedding_only and not args.clustermap_only:
        print(f"Saved {output_dir / f'{args.file_prefix}_spectral_k_scan.png'}")
    print(f"Saved {assignment_path}")
    if not args.embedding_only and not args.clustermap_only:
        print(
            f"Saved {output_dir / f'{args.file_prefix}_spectral_cluster_summary.csv'}"
        )
        print(f"Saved {output_dir / f'{args.file_prefix}_spectral_k_scan.csv'}")
    print(f"Saved {output_dir / f'{stem}_cluster_order.csv'}")
    print(f"sigma={sigma:.6f} main_k={args.main_k} silhouette={main_silhouette:.6f}")


if __name__ == "__main__":
    main()
