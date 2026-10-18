"""Render SEIRS trajectories, coefficient heatmaps, and the reproduction-ratio curve."""

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from scipy.integrate import solve_ivp
from ndtl.paths import ROOT
from .polynomial import BASIS_NAMES

LIGHT_BLUE, DARK_BLUE, LIGHT_RED, DARK_RED = "#73BAD6", "#02263E", "#EF4143", "#BF1E2E"


def reference_field(t, state):
    s, e, i = state
    r = 1 - (s + e + i)
    return [-0.9 * s * i + 0.15 * r, 0.9 * s * i - 0.1 * e, 0.1 * e - 0.2 * i]


def plot_trajectory(times, states, output, label):
    fig = plt.figure(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.patch.set_facecolor("none")
    for axis, bins in [(ax.xaxis, 3), (ax.yaxis, 3), (ax.zaxis, 2)]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("none")
        axis.set_major_locator(MaxNLocator(nbins=bins))
    ax.plot(*states.T, lw=2, color="black")
    ax.tick_params(pad=0, labelsize=16)
    fig.savefig(output / f"{label}_attractor.png", dpi=450)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for index, (name, color, style) in enumerate(
        [(r"$S$", LIGHT_RED, "-"), (r"$E$", DARK_BLUE, "--"), (r"$I$", LIGHT_BLUE, "-")]
    ):
        ax.plot(times, states[:, index], style, label=name, color=color, lw=2)
    ax.set_xlabel("Time", fontsize=20)
    ax.set_ylabel("Population", fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(labelsize=20, direction="in")
    ax.legend(prop={"size": 15}, loc="upper left")
    fig.savefig(output / f"{label}_states.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_ratios(coarse, fine, output):
    # Interior fine points exclude the endpoints shared with the coarse sweep.
    selections = [
        (coarse[:13], "Coarse", "o", LIGHT_BLUE),
        (fine[1:9], "Fine", "s", LIGHT_RED),
    ]
    fig, ax = plt.subplots(figsize=(8, 3))
    inset = inset_axes(ax, width="50%", height="50%", loc="upper right", borderpad=0)
    for rows, label, marker, color in selections:
        ax.plot(rows[:, 2], rows[:, 3], label=label, marker=marker, color=color)
        inset.plot(rows[:, 2], rows[:, 3], marker=marker, color=color)
    ax.set_xlabel(r"$\alpha_c : \alpha_s$", fontsize=20)
    ax.set_ylabel(r"$\beta : \gamma$", fontsize=20)
    ax.set_xscale("log")
    ax.legend()
    ax.tick_params(direction="in")
    inset.set_xscale("log")
    inset.set_xlim(0.02, 0.09)
    inset.set_xticks([0.02, 0.03, 0.04, 0.06, 0.08])
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.tick_params(labelsize=8, direction="in")
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.1")
    fig.savefig(output / "reproduction_ratio.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(path, indices, output):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    matrices = [
        np.array(
            [
                [
                    float(
                        next(
                            row
                            for row in rows
                            if int(row["index"]) == index
                            and row["component"] == component
                        )[name]
                    )
                    for name in BASIS_NAMES
                ]
                for component in ("E", "I")
            ]
        )
        for index in indices
    ]
    cmap = LinearSegmentedColormap.from_list(
        "seirs_coefficients", [LIGHT_BLUE, DARK_BLUE, LIGHT_RED, DARK_RED], N=256
    )
    fig, axes = plt.subplots(
        1, len(indices), figsize=(4 * len(indices), 4), squeeze=False
    )
    vmin, vmax = np.min(matrices), np.max(matrices)
    for ax, index, matrix in zip(axes[0], indices, matrices):
        image = ax.imshow(matrix, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(BASIS_NAMES)))
        ax.set_xticklabels(BASIS_NAMES, rotation=45, ha="right", fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["dE/dt", "dI/dt"], fontsize=10)
        ax.set_xlabel(f"Field {index}", fontsize=12)
    fig.colorbar(
        image, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.04, pad=0.04
    ).set_label("Coefficient", fontsize=12)
    fig.savefig(output / "coefficient_heatmap.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path, default=ROOT / "outputs/seirs/analysis"
    )
    parser.add_argument("--heatmap-indices", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratios = []
    for name in ("coarse", "fine"):
        data = np.loadtxt(
            args.analysis_dir / f"{name}_trajectory.csv", delimiter=",", skiprows=1
        )
        plot_trajectory(data[:, 0], data[:, 1:], args.out_dir, name)
        ratios.append(
            np.loadtxt(
                args.analysis_dir / f"{name}_ratios.csv",
                delimiter=",",
                skiprows=1,
                ndmin=2,
            )
        )
    plot_ratios(*ratios, args.out_dir)
    plot_coefficients(
        args.analysis_dir / "fine_coefficients.csv", args.heatmap_indices, args.out_dir
    )
    reference = solve_ivp(
        reference_field, (0, 50), [0.99, 0, 0.01], t_eval=np.linspace(0, 50, 5000)
    )
    plot_trajectory(reference.t, reference.y.T, args.out_dir, "reference")


if __name__ == "__main__":
    main()
