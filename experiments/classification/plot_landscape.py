#!/usr/bin/env python3
"""Similarity-density landscape for content-only distance matrices."""

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
from matplotlib import patheffects
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments/classification/results/systems52"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/classification/figures"
    / "dysts52_similarity_density_landscape"
)
PALETTE = [
    "#2a9d8f",
    "#e76f51",
    "#3a86ff",
    "#8338ec",
    "#f4a261",
    "#006d77",
    "#d62828",
    "#8ac926",
]


def remap_labels_by_position(labels: np.ndarray, coords: np.ndarray) -> np.ndarray:
    stats = []
    for old in sorted(set(int(x) for x in labels)):
        pts = coords[labels == old]
        center = pts.mean(axis=0)
        stats.append((float(center[0]), float(center[1]), old))
    stats.sort()
    mapping = {old: new for new, (_, _, old) in enumerate(stats, start=1)}
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def spectral_labels(
    affinity: np.ndarray, coords: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=50,
    ).fit_predict(affinity)
    return remap_labels_by_position(np.asarray(labels, dtype=int), coords)


def density_potential(
    affinity: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    no_self = np.asarray(affinity, dtype=float).copy()
    np.fill_diagonal(no_self, 0.0)
    density = no_self.sum(axis=1) / max(1, no_self.shape[0] - 1)
    potential = -np.log(density + eps)
    return density, potential


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    if len(values) <= 1:
        ranks.fill(0.0)
        return ranks
    ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
    return ranks


def idw_interpolate(
    coords: np.ndarray,
    values: np.ndarray,
    grid_size: int,
    padding: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin = coords.min(axis=0) - padding
    xmax, ymax = coords.max(axis=0) + padding
    xs = np.linspace(float(xmin), float(xmax), grid_size)
    ys = np.linspace(float(ymin), float(ymax), grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    diff = grid[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    weights = 1.0 / np.maximum(dist2, 1e-8) ** (0.5 * power)
    zz = np.sum(weights * values[None, :], axis=1) / np.sum(weights, axis=1)
    return xx, yy, zz.reshape(xx.shape)


def write_scores(
    path: Path,
    names: list[str],
    coords: np.ndarray,
    labels: np.ndarray,
    density: np.ndarray,
    potential: np.ndarray,
) -> None:
    density_rank = percentile_ranks(-density)
    potential_rank = percentile_ranks(potential)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "cluster",
                "embedding_1",
                "embedding_2",
                "similarity_density",
                "density_low_rank",
                "density_potential",
                "potential_high_rank",
            ]
        )
        for name, xy, label, rho, u, rho_rank, u_rank in zip(
            names, coords, labels, density, potential, density_rank, potential_rank
        ):
            writer.writerow(
                [
                    name,
                    int(label),
                    f"{xy[0]:.12g}",
                    f"{xy[1]:.12g}",
                    f"{rho:.12g}",
                    f"{rho_rank:.12g}",
                    f"{u:.12g}",
                    f"{u_rank:.12g}",
                ]
            )


def label_extreme_points(
    ax: plt.Axes,
    names: list[str],
    coords: np.ndarray,
    potential: np.ndarray,
    n_low: int,
    n_high: int,
    *,
    fontsize: float = 7.3,
    color: str = "#111111",
    stroke: bool = False,
) -> None:
    indices = list(np.argsort(potential)[:n_low]) + list(
        np.argsort(potential)[-n_high:]
    )
    seen: set[int] = set()
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_margin = 0.10 * max(1e-12, float(x_max - x_min))
    y_margin = 0.10 * max(1e-12, float(y_max - y_min))
    for idx in indices:
        i = int(idx)
        if i in seen:
            continue
        seen.add(i)
        dx = 6
        dy = 6 if potential[i] <= np.median(potential) else -10
        ha = "left"
        va = "bottom" if dy >= 0 else "top"
        if coords[i, 0] >= x_max - x_margin:
            dx = -6
            ha = "right"
        elif coords[i, 0] <= x_min + x_margin:
            dx = 6
            ha = "left"
        if coords[i, 1] >= y_max - y_margin:
            dy = -10
            va = "top"
        elif coords[i, 1] <= y_min + y_margin:
            dy = 6
            va = "bottom"
        if names[i] == "rayleigh_benard":
            dx = 6
            dy = -18
            ha = "left"
            va = "top"
        text = ax.annotate(
            display_name(names[i]),
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            ha=ha,
            va=va,
            zorder=8,
        )
        if stroke:
            text.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=2.2, foreground="#111111", alpha=0.88
                    )
                ]
            )


def plot_contour(
    path: Path,
    names: list[str],
    coords: np.ndarray,
    labels: np.ndarray,
    potential: np.ndarray,
    sigma: float,
    dataset_label: str,
    dpi: int,
    grid_size: int,
    paper_contour: bool,
) -> None:
    xx, yy, zz = idw_interpolate(
        coords, potential, grid_size=grid_size, padding=0.08, power=2.0
    )
    levels = np.linspace(float(np.nanmin(potential)), float(np.nanmax(potential)), 18)
    fig_size = (11.2, 10.0) if paper_contour else (12.6, 10.0)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", alpha=0.88)
    contour = ax.contour(
        xx, yy, zz, levels=levels, colors="#222222", linewidths=0.55, alpha=0.45
    )
    ax.clabel(contour, inline=True, fontsize=8.8 if paper_contour else 7, fmt="%.2f")

    for label in sorted(set(int(x) for x in labels)):
        pts = coords[labels == label]
        color = PALETTE[(label - 1) % len(PALETTE)]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=72,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            label=f"cluster {label}",
            zorder=6,
        )

    if paper_contour:
        label_extreme_points(
            ax,
            names,
            coords,
            potential,
            n_low=10,
            n_high=8,
            fontsize=14.0,
            color="#ffffff",
            stroke=True,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=10.5, labelrotation=90, pad=2)
        for tick in ax.get_yticklabels():
            tick.set_horizontalalignment("center")
            tick.set_verticalalignment("center")
    else:
        label_extreme_points(ax, names, coords, potential, n_low=10, n_high=8)
        cbar = fig.colorbar(filled, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(
            "density potential U = -log(mean RBF affinity); lower = dense basin"
        )
        ax.set_title(
            f"{dataset_label} Content-Only Similarity-Density Landscape\nheight = -log(mean RBF affinity to other systems)"
        )
        ax.text(
            0.01,
            0.01,
            f"sigma={sigma:.3f}; surface is IDW interpolation over the displayed grid",
            transform=ax.transAxes,
            fontsize=8,
            color="#333333",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )
        ax.set_xlabel("spectral embedding 1")
        ax.set_ylabel("spectral embedding 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.35, linewidth=0.7)
    if not paper_contour:
        ax.legend(frameon=True, fontsize=7.8, loc="upper right", ncol=1)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_surface(
    path: Path,
    names: list[str],
    coords: np.ndarray,
    labels: np.ndarray,
    potential: np.ndarray,
    sigma: float,
    dataset_label: str,
    dpi: int,
    paper_surface: bool,
) -> None:
    xx, yy, zz = idw_interpolate(
        coords, potential, grid_size=165, padding=0.08, power=2.0
    )
    fig = plt.figure(
        figsize=(13.2, 9.8) if paper_surface else (12.8, 9.4), constrained_layout=True
    )
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        xx, yy, zz, cmap="viridis", alpha=0.84, linewidth=0, antialiased=True
    )
    point_size = 68 if paper_surface else 46
    point_linewidth = 0.9 if paper_surface else 0.7
    text_fontsize = 12.5 if paper_surface else 7.0
    for label in sorted(set(int(x) for x in labels)):
        pts = coords[labels == label]
        vals = potential[labels == label]
        color = PALETTE[(label - 1) % len(PALETTE)]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            vals,
            s=point_size,
            color=color,
            edgecolor="white",
            linewidth=point_linewidth,
            depthshade=False,
            label=f"cluster {label}",
        )

    for idx in list(np.argsort(potential)[:7]) + list(np.argsort(potential)[-7:]):
        i = int(idx)
        text = ax.text(
            coords[i, 0],
            coords[i, 1],
            potential[i] + 0.03,
            display_name(names[i]),
            fontsize=text_fontsize,
            color="#ffffff" if paper_surface else "#111111",
        )
        if paper_surface:
            text.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=2.6, foreground="#111111", alpha=0.92
                    )
                ]
            )

    if not paper_surface:
        ax.set_title(f"3D {dataset_label} Similarity-Density Landscape", pad=12)
        ax.set_xlabel("spectral embedding 1")
        ax.set_ylabel("spectral embedding 2")
        ax.set_zlabel("U = -log(mean RBF affinity)")
    else:
        ax.set_xlabel("Spectral embedding coordinate 1", fontsize=16, labelpad=14)
        ax.set_ylabel("Spectral embedding coordinate 2", fontsize=16, labelpad=14)
        ax.set_zlabel("U = -log(mean RBF affinity)", fontsize=14, labelpad=12)
        ax.tick_params(axis="x", labelsize=13, pad=2)
        ax.tick_params(axis="y", labelsize=13, pad=2)
        ax.tick_params(axis="z", labelsize=12, pad=2)
    ax.view_init(elev=32, azim=-58)
    if paper_surface:
        cbar = fig.colorbar(
            surface, ax=ax, fraction=0.023, pad=0.055, shrink=0.62, aspect=28
        )
        cbar.ax.tick_params(labelsize=11)
        ax.legend(frameon=True, fontsize=11, markerscale=1.25, loc="upper left")
    else:
        cbar = fig.colorbar(surface, ax=ax, fraction=0.035, pad=0.08)
        cbar.set_label(f"density potential; sigma={sigma:.3f}")
        ax.legend(frameon=True, fontsize=7, loc="upper left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_extreme_scores(
    path: Path,
    names: list[str],
    density: np.ndarray,
    potential: np.ndarray,
    dataset_label: str,
    dpi: int,
    n: int,
) -> None:
    basin_idx = list(np.argsort(potential)[:n])
    ridge_idx = list(np.argsort(potential)[-n:][::-1])
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4), constrained_layout=True)
    for ax, indices, title, values, color in [
        (axes[0], basin_idx, "lowest potential / densest basins", potential, "#2a9d8f"),
        (
            axes[1],
            ridge_idx,
            "highest potential / sparsest ridges",
            potential,
            "#d62828",
        ),
    ]:
        labels = [display_name(names[i]) for i in indices]
        vals = [float(values[i]) for i in indices]
        y = np.arange(len(indices))
        ax.barh(y, vals, color=color, alpha=0.86)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.7)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("density potential U")
        for yy, val in zip(y, vals):
            ax.text(val, yy, f" {val:.3f}", va="center", fontsize=8.0)
    fig.suptitle(f"{dataset_label} Similarity-Density Potential Extremes")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--dataset-label", type=str, default="DYSTS52")
    parser.add_argument("--file-prefix", type=str, default="dysts52")
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--extreme-count", type=int, default=14)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--paper-contour", action="store_true")
    parser.add_argument("--paper-surface", action="store_true")
    parser.add_argument("--contour-only", action="store_true")
    parser.add_argument("--surface-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contour_only and args.surface_only:
        raise ValueError("--contour-only and --surface-only cannot be used together")
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
    labels = spectral_labels(affinity, coords, args.clusters, args.random_state)
    density, potential = density_potential(affinity, args.eps)

    prefix = args.file_prefix
    contour_path = output_dir / f"{prefix}_similarity_density_landscape_contour.png"
    surface_path = output_dir / f"{prefix}_similarity_density_landscape_surface.png"
    extremes_path = output_dir / f"{prefix}_similarity_density_landscape_extremes.png"
    scores_path = output_dir / f"{prefix}_similarity_density_landscape_scores.csv"

    if not args.surface_only:
        plot_contour(
            contour_path,
            names,
            coords,
            labels,
            potential,
            sigma,
            args.dataset_label,
            args.dpi,
            args.grid_size,
            args.paper_contour,
        )
    if not args.contour_only:
        plot_surface(
            surface_path,
            names,
            coords,
            labels,
            potential,
            sigma,
            args.dataset_label,
            args.dpi,
            args.paper_surface,
        )
        if not args.surface_only:
            plot_extreme_scores(
                extremes_path,
                names,
                density,
                potential,
                args.dataset_label,
                args.dpi,
                n=min(args.extreme_count, len(names)),
            )
    if not args.surface_only:
        write_scores(scores_path, names, coords, labels, density, potential)

    if not args.surface_only:
        print(f"Saved {contour_path}")
    if not args.contour_only:
        print(f"Saved {surface_path}")
        if not args.surface_only:
            print(f"Saved {extremes_path}")
    if not args.surface_only:
        print(f"Saved {scores_path}")
    print(
        f"sigma={sigma:.6f} potential_min={float(np.min(potential)):.6f} potential_max={float(np.max(potential)):.6f}"
    )


if __name__ == "__main__":
    main()
