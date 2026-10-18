from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import integrate_rhs
from .holling import HP_FALLBACK_MODULES, HollingParams, holling_rhs_ecology

COLORS = {"P": "#276FBF", "H": "#2A9D8F", "C": "#D95F02"}
LATE_COLOR = "#303030"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fitted_alpha_params(report_path: Path, line: str, alpha: float):
    report = load_json(report_path)
    rows = report["lines"][line]["rows"]
    row = next(row for row in rows if abs(row["alpha_b"] - alpha) < 1e-12)
    return HollingParams(**row["holling_fit"]["params"])


def downsample_indices(n: int, max_points: int = 2600):
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=int)


def padded_limits(values, pad_frac: float = 0.08, min_span: float = 0.05):
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    center = 0.5 * (lo + hi)
    span = max(hi - lo, min_span)
    pad = pad_frac * span
    return center - 0.5 * span - pad, center + 0.5 * span + pad


def phase_limits(traj: np.ndarray):
    min_spans = [0.12, 0.08, 0.6]
    return [
        padded_limits(traj[:, i], pad_frac=0.10, min_span=min_spans[i])
        for i in range(3)
    ]


def style_3d_axis(ax):
    ax.tick_params(axis="both", which="major", labelsize=8.5, pad=0)
    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = -1
    ax.zaxis.labelpad = -4
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.xaxis.pane.set_edgecolor("#D0D0D0")
    ax.yaxis.pane.set_edgecolor("#D0D0D0")
    ax.zaxis.pane.set_edgecolor("#D0D0D0")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.83, 0.83, 0.83, 0.65)
        axis._axinfo["grid"]["linewidth"] = 0.45


def plot_paper_system(
    out_base: Path,
    title: str,
    params: HollingParams,
    init_state,
    steps: int,
    dt: float,
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

    fig = plt.figure(figsize=(7.15, 2.55), dpi=360)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.48], wspace=0.43)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_c = ax_ts.twinx()

    ax3d.plot(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        color=LATE_COLOR,
        lw=0.72,
    )
    xlim, ylim, zlim = phase_limits(traj)
    ax3d.set_xlim(xlim)
    ax3d.set_ylim(ylim)
    ax3d.set_zlim(zlim)
    ax3d.set_xlabel("P", fontsize=11)
    ax3d.set_ylabel("H", fontsize=11)
    ax3d.set_zlabel("C", fontsize=11)
    ax3d.view_init(elev=24, azim=-58)
    style_3d_axis(ax3d)

    line_p = ax_ts.plot(shown_t, shown[:, 0], color=COLORS["P"], lw=0.9, label="P")[0]
    line_h = ax_ts.plot(shown_t, shown[:, 1], color=COLORS["H"], lw=0.9, label="H")[0]
    line_c = ax_c.plot(shown_t, shown[:, 2], color=COLORS["C"], lw=0.9, label="C")[0]

    ax_ts.set_xlim(float(t[0]), float(t[-1]))
    ax_ts.set_ylim(padded_limits(traj[:, :2].reshape(-1), pad_frac=0.10, min_span=0.08))
    ax_c.set_ylim(padded_limits(traj[:, 2], pad_frac=0.10, min_span=0.5))
    ax_ts.set_xlabel("time", fontsize=9, labelpad=1)
    ax_ts.set_ylabel("P, H", fontsize=11, color="#404040", labelpad=2)
    ax_c.set_ylabel("C", fontsize=11, color=COLORS["C"], labelpad=2)
    ax_ts.tick_params(axis="both", labelsize=8.5, pad=1)
    ax_c.tick_params(axis="y", labelsize=8.5, pad=1, colors=COLORS["C"])
    ax_ts.grid(alpha=0.22, lw=0.5)
    ax_ts.legend(
        [line_p, line_h, line_c],
        ["P", "H", "C"],
        loc="upper center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.50, 1.03),
        handlelength=1.6,
        columnspacing=1.1,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "status": status,
        "min": traj.min(axis=0).tolist(),
        "max": traj.max(axis=0).tolist(),
        "params": params.to_dict(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make compact paper figures for HP/Holling transient trajectories."
    )
    parser.add_argument("--report", default="experiments/hp/results/main/report.json")
    parser.add_argument("--out-dir", default="outputs/hp/figures")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--init", type=float, nargs=3, default=(0.5, 0.2, 8.0))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    systems = [
        ("hp_stable_parent_paper", "Stable parent", HP_FALLBACK_MODULES["stable"]),
        ("hp_chaotic_parent_paper", "Chaotic parent", HP_FALLBACK_MODULES["chaotic"]),
        (
            "hp_alpha0p50_holling_projection_paper",
            "NDTL alpha=0.50",
            fitted_alpha_params(report_path, "stable_to_chaotic", 0.5),
        ),
    ]
    manifest = []
    for slug, title, params in systems:
        manifest.append(
            plot_paper_system(
                out_dir / slug, title, params, args.init, args.steps, args.dt
            )
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
