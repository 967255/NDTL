"""Fit sparse SEIRS models and export coefficients, reproduction ratios, and trajectories."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from ndtl.paths import ROOT
from .polynomial import BASIS_NAMES, fit_field, load_sweep


def analyze_sweep(path, output, name, trajectory_index, alpha=0.005):
    fields = load_sweep(path)
    if not 0 <= trajectory_index < len(fields):
        raise ValueError(
            f"{name} trajectory index {trajectory_index} is outside {len(fields)} fields"
        )
    config_path = Path(path).parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    weights = config.get("content_weights")
    if weights is None:
        weights = json.loads(
            (ROOT / f"experiments/seirs/configs/{name}.json").read_text()
        )["content_weights"]
    if len(weights) != len(fields):
        raise ValueError("weight count does not match the input fields")
    style_weight = config.get("style_weight", 0.01)
    if style_weight <= 0:
        raise ValueError("the reproduction-ratio plot requires positive style weight")
    coefficient_rows, ratios = [], []
    for index, field in enumerate(fields[:, 0]):
        model = fit_field(field, alpha)
        for component, values in zip(("S", "E", "I"), model.coefficient_matrix()):
            coefficient_rows.append([index, component, *values])
        ratios.append(
            [
                index,
                weights[index],
                weights[index] / style_weight,
                model.reproduction_ratio,
            ]
        )
        if index == trajectory_index:
            solution = solve_ivp(
                model.vector_field,
                (0, 50),
                [0.99, 0.0, 0.01],
                t_eval=np.linspace(0, 50, 5000),
            )
            np.savetxt(
                output / f"{name}_trajectory.csv",
                np.column_stack((solution.t, solution.y.T)),
                delimiter=",",
                header="time,S,E,I",
                comments="",
            )
            trajectory = {
                "index": index,
                "success": bool(solution.success),
                "message": solution.message,
            }
    with (output / f"{name}_coefficients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "component", *BASIS_NAMES])
        writer.writerows(coefficient_rows)
    with (output / f"{name}_ratios.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["index", "content_weight", "content_style_ratio", "reproduction_ratio"]
        )
        writer.writerows(ratios)
    return {
        "input": str(path),
        "fields": len(fields),
        "lasso_alpha": alpha,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse", type=Path, default=ROOT / "outputs/seirs/coarse/vecs.npy"
    )
    parser.add_argument(
        "--fine", type=Path, default=ROOT / "outputs/seirs/fine/vecs.npy"
    )
    parser.add_argument("--coarse-index", type=int, default=6)
    parser.add_argument("--fine-index", type=int, default=9)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.lasso_alpha < 0 or not np.isfinite(args.lasso_alpha):
        parser.error("--lasso-alpha must be finite and nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        name: analyze_sweep(path, args.out_dir, name, index, args.lasso_alpha)
        for name, path, index in [
            ("coarse", args.coarse, args.coarse_index),
            ("fine", args.fine, args.fine_index),
        ]
    }
    (args.out_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
