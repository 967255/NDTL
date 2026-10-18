from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import TrilinearVectorField, integrate_rhs, lyapunov_spectrum
from .holling import HP_FALLBACK_MODULES, HollingParams, holling_rhs_ecology


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def initial_conditions():
    ps = [0.2, 0.5, 0.9]
    hs = [0.05, 0.18, 0.36]
    cs = [7.5, 9.5, 13.0]
    return [np.array([p, h, c], dtype=np.float64) for p in ps for h in hs for c in cs]


def local_peaks(x: np.ndarray):
    if x.size < 3:
        return np.array([], dtype=np.int64)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1


def trajectory_stats(traj: np.ndarray, status: str, dt: float):
    if traj.size == 0:
        return {
            "status": status,
            "persistent": False,
            "min_tail": [float("nan")] * 3,
            "mean_tail": [float("nan")] * 3,
            "std_tail": [float("nan")] * 3,
            "max_tail": [float("nan")] * 3,
            "peak_count_c": 0,
            "peak_interval_mean": None,
            "peak_interval_cv": None,
            "peak_height_mean": None,
            "peak_height_cv": None,
        }
    c = traj[:, 2]
    peaks = local_peaks(c)
    intervals = (
        np.diff(peaks).astype(np.float64) * dt if peaks.size >= 2 else np.array([])
    )
    heights = c[peaks] if peaks.size else np.array([])

    def cv(values):
        if values.size < 2:
            return None
        mean = float(np.mean(values))
        return None if abs(mean) < 1e-12 else float(np.std(values) / abs(mean))

    return {
        "status": status,
        "persistent": bool(status == "normal" and np.min(traj) > 1e-3),
        "min_tail": list(map(float, np.min(traj, axis=0))),
        "mean_tail": list(map(float, np.mean(traj, axis=0))),
        "std_tail": list(map(float, np.std(traj, axis=0))),
        "max_tail": list(map(float, np.max(traj, axis=0))),
        "peak_count_c": int(peaks.size),
        "peak_interval_mean": (
            None if intervals.size == 0 else float(np.mean(intervals))
        ),
        "peak_interval_cv": cv(intervals),
        "peak_height_mean": None if heights.size == 0 else float(np.mean(heights)),
        "peak_height_cv": cv(heights),
    }


def classify_multistart(per_ic: list[dict]):
    persistent = [row["stats"]["persistent"] for row in per_ic]
    peak_counts = np.array(
        [row["stats"]["peak_count_c"] for row in per_ic], dtype=np.float64
    )
    std_norms = np.array(
        [np.linalg.norm(row["stats"]["std_tail"]) for row in per_ic], dtype=np.float64
    )
    interval_cvs = [
        row["stats"]["peak_interval_cv"]
        for row in per_ic
        if row["stats"]["peak_interval_cv"] is not None
    ]
    height_cvs = [
        row["stats"]["peak_height_cv"]
        for row in per_ic
        if row["stats"]["peak_height_cv"] is not None
    ]

    interval_means = [
        row["stats"]["peak_interval_mean"]
        for row in per_ic
        if row["stats"]["peak_interval_mean"] is not None
    ]
    height_means = [
        row["stats"]["peak_height_mean"]
        for row in per_ic
        if row["stats"]["peak_height_mean"] is not None
    ]

    return {
        "n_initial_conditions": len(per_ic),
        "n_persistent": int(np.sum(persistent)),
        "n_oscillatory_by_c_peaks": int(np.sum(peak_counts >= 5)),
        "std_norm_median": float(np.nanmedian(std_norms)),
        "peak_count_c_median": float(np.median(peak_counts)),
        "within_trajectory_peak_interval_cv_median": (
            None if not interval_cvs else float(np.median(interval_cvs))
        ),
        "within_trajectory_peak_height_cv_median": (
            None if not height_cvs else float(np.median(height_cvs))
        ),
        "across_ic_peak_interval_mean_cv": coefficient_of_variation(interval_means),
        "across_ic_peak_height_mean_cv": coefficient_of_variation(height_means),
    }


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    mean = float(np.mean(values))
    return None if abs(mean) < 1e-12 else float(np.std(values) / abs(mean))


def run_system(name: str, rhs, steps: int, keep: int, dt: float, lyap_count: int):
    ics = initial_conditions()
    per_ic = []
    for idx, ic in enumerate(ics):
        traj, status = integrate_rhs(rhs, init_state=ic, steps=steps, keep=keep, dt=dt)
        per_ic.append(
            {
                "ic_index": idx,
                "initial_condition": list(map(float, ic)),
                "stats": trajectory_stats(traj, status, dt),
            }
        )

    lyapunov_rows = []
    for idx, ic in enumerate(ics[:lyap_count]):
        lyap = lyapunov_spectrum(
            rhs,
            init_state=ic,
            dt=0.03,
            transient_steps=8000,
            measure_steps=16000,
        )
        lyapunov_rows.append(
            {
                "ic_index": idx,
                "initial_condition": list(map(float, ic)),
                "lyapunov": None if lyap is None else list(map(float, lyap)),
            }
        )

    return {
        "name": name,
        "summary": classify_multistart(per_ic),
        "per_initial_condition": per_ic,
        "lyapunov_checks": lyapunov_rows,
    }


def make_holling_params(params_dict: dict, name: str):
    fields = {
        k: params_dict[k]
        for k in HollingParams.__dataclass_fields__
        if k in params_dict
    }
    fields["name"] = name
    return HollingParams(**fields)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-start validation for HP/Holling ecology NDTL results."
    )
    parser.add_argument(
        "--report", default="outputs/hp/main/fixed02/main64/report.json"
    )
    parser.add_argument("--line", default="stable_to_chaotic")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/hp/validation/multistart.json")
    parser.add_argument("--steps", type=int, default=90000)
    parser.add_argument("--keep", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--lyap-count", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report)
    report = load_json(report_path)
    domain = report["domain"]
    rows = report["lines"][args.line]["rows"]
    row = next(row for row in rows if abs(row["alpha_b"] - args.alpha) < 1e-12)

    field_path = report_path.parent / args.line / f"fused_alpha_{args.alpha:.2f}.npz"
    with np.load(field_path) as data:
        field_unit = data["field_unit"]
    fused_rhs = TrilinearVectorField(field_unit, domain=domain)

    fitted_params = make_holling_params(
        row["holling_fit"]["params"], "alpha0p50_holling_fit"
    )
    fitted_rhs = lambda state: holling_rhs_ecology(state, fitted_params)

    systems = {
        "stable_parent_exact": lambda state: holling_rhs_ecology(
            state, HP_FALLBACK_MODULES["stable"]
        ),
        "chaotic_parent_exact": lambda state: holling_rhs_ecology(
            state, HP_FALLBACK_MODULES["chaotic"]
        ),
        "fused_alpha0p50_ndtl_field": fused_rhs,
        "fused_alpha0p50_holling_fit": fitted_rhs,
    }

    output = {
        "report": str(report_path),
        "line": args.line,
        "alpha": args.alpha,
        "domain": domain,
        "initial_condition_grid": [list(map(float, ic)) for ic in initial_conditions()],
        "integration": {"steps": args.steps, "keep": args.keep, "dt": args.dt},
        "systems": [],
    }
    for name, rhs in systems.items():
        output["systems"].append(
            run_system(name, rhs, args.steps, args.keep, args.dt, args.lyap_count)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
