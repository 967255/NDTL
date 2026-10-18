from __future__ import annotations

import argparse
from .plot import fitted_alpha_params, padded_limits
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import integrate_rhs
from .holling import HP_FALLBACK_MODULES, HollingParams, holling_rhs_ecology

COLORS = {"P": "#276FBF", "H": "#2A9D8F", "C": "#D95F02"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def downsample_indices(n: int, max_points: int = 2400):
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def phase_limits(traj: np.ndarray):
    min_spans = [0.12, 0.08, 0.6]
    return [
        padded_limits(traj[:, i], pad_frac=0.10, min_span=min_spans[i])
        for i in range(3)
    ]


def plot_system(
    path: Path, title: str, params: HollingParams, init_state, steps: int, dt: float
):
    rhs = lambda state: holling_rhs_ecology(state, params)
    traj, status = integrate_rhs(
        rhs, init_state=init_state, dt=dt, steps=steps, keep=steps
    )
    if status != "normal" or traj.size == 0:
        raise RuntimeError(f"{title} integration failed with status={status}")

    t = np.arange(traj.shape[0], dtype=np.float64) * dt
    idx = downsample_indices(traj.shape[0])
    shown = traj[idx]
    shown_t = t[idx]

    fig = plt.figure(figsize=(12.8, 7.4), dpi=190)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.55], wspace=0.24, hspace=0.34)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    split = max(3, int(0.18 * shown.shape[0]))
    ax3d.plot(
        shown[:split, 0],
        shown[:split, 1],
        shown[:split, 2],
        color="#D95F02",
        lw=0.8,
        label="early",
    )
    ax3d.plot(
        shown[split - 1 :, 0],
        shown[split - 1 :, 1],
        shown[split - 1 :, 2],
        color="#303030",
        lw=0.65,
        label="later",
    )
    ax3d.scatter(
        shown[0, 0], shown[0, 1], shown[0, 2], s=34, color="#2A9D8F", label="start"
    )
    ax3d.scatter(
        shown[-1, 0], shown[-1, 1], shown[-1, 2], s=34, color="#C1121F", label="end"
    )
    (xlim, ylim, zlim) = phase_limits(traj)
    ax3d.set_xlim(xlim)
    ax3d.set_ylim(ylim)
    ax3d.set_zlim(zlim)
    ax3d.set_xlabel("P")
    ax3d.set_ylabel("H")
    ax3d.set_zlabel("C")
    ax3d.view_init(elev=25, azim=-58)
    ax3d.set_title("phase trajectory", fontsize=11)
    ax3d.legend(loc="upper left", fontsize=8)

    for ax, label, col in zip(axes, ["P", "H", "C"], [0, 1, 2]):
        ax.plot(shown_t, shown[:, col], color=COLORS[label], lw=0.85)
        ax.scatter(t[0], traj[0, col], s=12, color="#2A9D8F", zorder=3)
        ax.scatter(t[-1], traj[-1, col], s=12, color="#C1121F", zorder=3)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(
            padded_limits(traj[:, col], pad_frac=0.10, min_span=[0.08, 0.04, 0.5][col])
        )
        ax.set_ylabel(label)
        ax.grid(alpha=0.22, lw=0.5)
    axes[-1].set_xlabel("time")

    fig.suptitle(
        f"{title} | status={status} | init={tuple(init_state)}", fontsize=13, y=0.985
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "status": status,
        "min": traj.min(axis=0).tolist(),
        "max": traj.max(axis=0).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot transient-preserving trajectories for three HP/Holling systems."
    )
    parser.add_argument("--report", default="experiments/hp/results/main/report.json")
    parser.add_argument("--out-dir", default="outputs/hp/transients")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--init", type=float, nargs=3, default=(0.5, 0.2, 8.0))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    systems = [
        (
            "stable_parent",
            "stable parent exact Holling system",
            HP_FALLBACK_MODULES["stable"],
        ),
        (
            "chaotic_parent",
            "chaotic parent exact Holling system",
            HP_FALLBACK_MODULES["chaotic"],
        ),
        (
            "alpha0p50_holling_projection",
            "NDTL alpha=0.50 projected-back Holling system",
            fitted_alpha_params(report_path, "stable_to_chaotic", 0.5),
        ),
    ]
    manifest = []
    for slug, title, params in systems:
        result = plot_system(
            out_dir / f"{slug}_transient_phase_time.png",
            title,
            params,
            args.init,
            args.steps,
            args.dt,
        )
        result["params"] = params.to_dict()
        manifest.append(result)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
