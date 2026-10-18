from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import TrilinearVectorField, integrate_rhs
from .holling import HP_FALLBACK_MODULES, holling_rhs_ecology

SPECIES_COLORS = {"P": "#276FBF", "H": "#2A9D8F", "C": "#D95F02"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stride_view(traj: np.ndarray, max_points: int):
    if traj.shape[0] <= max_points:
        return traj
    return traj[:: max(1, traj.shape[0] // max_points)]


def set_phase_limits(ax, domain):
    ax.set_xlim(domain[0])
    ax.set_ylim(domain[1])
    ax.set_zlim(domain[2])
    ax.set_xlabel("P")
    ax.set_ylabel("H")
    ax.set_zlabel("C")
    ax.view_init(elev=24, azim=-58)


def plot_row(ax_phase, ax_time, traj: np.ndarray, title: str, domain, dt: float):
    shown = stride_view(traj, 5000)
    ax_phase.plot(shown[:, 0], shown[:, 1], shown[:, 2], lw=0.65, color="#303030")
    ax_phase.scatter(shown[-1:, 0], shown[-1:, 1], shown[-1:, 2], s=10, color="#C1121F")
    set_phase_limits(ax_phase, domain)
    ax_phase.set_title(title, fontsize=9)

    t = np.arange(traj.shape[0], dtype=np.float64) * dt
    shown_idx = np.arange(traj.shape[0])
    if traj.shape[0] > 5000:
        shown_idx = shown_idx[:: max(1, traj.shape[0] // 5000)]
    ax_time.plot(
        t[shown_idx], traj[shown_idx, 0], color=SPECIES_COLORS["P"], lw=0.8, label="P"
    )
    ax_time.plot(
        t[shown_idx], traj[shown_idx, 1], color=SPECIES_COLORS["H"], lw=0.8, label="H"
    )
    ax_time.plot(
        t[shown_idx], traj[shown_idx, 2], color=SPECIES_COLORS["C"], lw=0.8, label="C"
    )
    ax_time.set_title(title, fontsize=9)
    ax_time.set_xlabel("time")
    ax_time.set_ylabel("state")
    ax_time.grid(alpha=0.22, lw=0.5)


def parent_trajectories(steps: int, keep: int, dt: float):
    out = {}
    for label, params in HP_FALLBACK_MODULES.items():
        rhs = lambda state, p=params: holling_rhs_ecology(state, p)
        traj, status = integrate_rhs(rhs, steps=steps, keep=keep, dt=dt)
        out[label] = {"traj": traj, "status": status, "params": params}
    return out


def plot_parents(out_path: Path, domain, steps: int, keep: int, dt: float):
    labels = ["stable", "periodic", "chaotic"]
    trajectories = parent_trajectories(steps=steps, keep=keep, dt=dt)
    fig = plt.figure(figsize=(12.5, 10.0), dpi=180)
    for i, label in enumerate(labels):
        params = trajectories[label]["params"]
        status = trajectories[label]["status"]
        title = f"parent {label} | {params.name} | status={status}"
        ax_phase = fig.add_subplot(len(labels), 2, 2 * i + 1, projection="3d")
        ax_time = fig.add_subplot(len(labels), 2, 2 * i + 2)
        plot_row(ax_phase, ax_time, trajectories[label]["traj"], title, domain, dt)
        if i == 0:
            ax_time.legend(loc="upper right", fontsize=8, ncol=3)
    fig.suptitle("HP/Holling parent systems", y=0.995, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_fused_trajectory(npz_path: Path, domain, steps: int, keep: int, dt: float):
    with np.load(npz_path) as data:
        field_unit = data["field_unit"]
    rhs = TrilinearVectorField(field_unit, domain=domain)
    traj, status = integrate_rhs(rhs, steps=steps, keep=keep, dt=dt)
    return traj, status


def plot_fusion_line(
    report_path: Path,
    line_name: str,
    out_path: Path,
    steps: int,
    keep: int,
    dt: float,
):
    report = load_json(report_path)
    domain = report["domain"]
    line = report["lines"][line_name]
    rows = line["rows"]
    fig = plt.figure(figsize=(13.0, 16.0), dpi=180)
    for i, row in enumerate(rows):
        alpha = row["alpha_b"]
        npz_path = report_path.parent / line_name / f"fused_alpha_{alpha:.2f}.npz"
        traj, status = load_fused_trajectory(
            npz_path, domain=domain, steps=steps, keep=keep, dt=dt
        )
        dyn = row["dynamics"]
        fit = row["holling_fit"]
        title = (
            f"{report['domain_mode']} | {line_name} | alpha={alpha:.2f} | "
            f"status={status} | persistent={dyn['persistent']} | fit={fit['rel_rmse']:.3g}"
        )
        ax_phase = fig.add_subplot(len(rows), 2, 2 * i + 1, projection="3d")
        ax_time = fig.add_subplot(len(rows), 2, 2 * i + 2)
        plot_row(ax_phase, ax_time, traj, title, domain, dt)
        if i == 0:
            ax_time.legend(loc="upper right", fontsize=8, ncol=3)
    fig.suptitle(
        f"Fusion trajectory overview: {report['domain_mode']} / {line_name}",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_parent_overlay(out_path: Path, domain, steps: int, keep: int, dt: float):
    trajectories = parent_trajectories(steps=steps, keep=keep, dt=dt)
    fig = plt.figure(figsize=(12.0, 5.2), dpi=180)
    ax_phase = fig.add_subplot(1, 2, 1, projection="3d")
    ax_time = fig.add_subplot(1, 2, 2)
    colors = {"stable": "#1B9E77", "periodic": "#7570B3", "chaotic": "#D95F02"}
    for label, item in trajectories.items():
        traj = item["traj"]
        shown = stride_view(traj, 5000)
        ax_phase.plot(
            shown[:, 0],
            shown[:, 1],
            shown[:, 2],
            lw=0.8,
            label=label,
            color=colors[label],
        )
        t = np.arange(traj.shape[0], dtype=np.float64) * dt
        shown_idx = np.arange(traj.shape[0])
        if traj.shape[0] > 4000:
            shown_idx = shown_idx[:: max(1, traj.shape[0] // 4000)]
        ax_time.plot(
            t[shown_idx],
            traj[shown_idx, 2],
            lw=0.8,
            label=f"{label} C",
            color=colors[label],
        )
    set_phase_limits(ax_phase, domain)
    ax_phase.legend(fontsize=8)
    ax_time.set_xlabel("time")
    ax_time.set_ylabel("C")
    ax_time.grid(alpha=0.22, lw=0.5)
    ax_time.legend(fontsize=8)
    fig.suptitle(
        "Parent overlay: phase-space and carnivore time series", y=0.99, fontsize=13
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize HP/Holling NDTL trajectories."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/hp/main/fixed02/main64/report.json"),
    )
    parser.add_argument("--out-dir", default="outputs/hp/diagnostics")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--keep", type=int, default=9000)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = args.report
    out_dir = Path(args.out_dir)
    report = load_json(report_path)
    domain = report["domain"]
    plot_parents(
        out_dir / "parents_phase_time_grid.png", domain, args.steps, args.keep, args.dt
    )
    plot_parent_overlay(
        out_dir / "parents_overlay.png", domain, args.steps, args.keep, args.dt
    )
    for line in report["lines"]:
        plot_fusion_line(
            report_path,
            line,
            out_dir / f"{line}_phase_time_grid.png",
            args.steps,
            args.keep,
            args.dt,
        )
    print(f"wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
