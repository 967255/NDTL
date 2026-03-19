import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
# ./results_lirt/LIRT_pre_noise_lorenz_chen_lv_ori/lorenz
RUN_DIR = "./results_lirt/LIRT_pre_noise_lorenz_chen_lv_ori/lorenz"

TARGET_LORENZ_PARAMS = {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 8.0 / 3.0,
}

OUTPUT_JSON = os.path.join(RUN_DIR, "eval_lorenz.json")

# Visualization (MVP)
VIZ_CONFIG = {
    "enabled": True,
    "use_stat": "mean",            # "mean" or "median"
    "x_log_scale": True,           # future-proof for log-spaced eps
    "show_error_band": True,       # std band if mean, IQR band if median
    "dpi": 220,
    "save_dir": os.path.join(RUN_DIR, "figures"),
    "make_heatmap_deltaD": True,   # OK even in MVP
}


# =========================
# Helpers
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_xyz_grids(grid_shape, grid_limits):
    """
    Build meshgrid with indexing='ij' to match [D,H,W] ordering.
    grid_shape: [D,H,W]
    grid_limits: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
    """
    (nx, ny, nz) = grid_shape
    (xlim, ylim, zlim) = grid_limits

    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float64)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float64)
    zs = np.linspace(zlim[0], zlim[1], nz, dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


def load_vec_from_npz(npz_path):
    """
    Supports both:
      - baseline/original_vec.npz with key 'vec'
      - recon files with key 'vec_hat'
    Returns vec ndarray with shape [D,H,W,3]
    """
    data = np.load(npz_path)
    if "vec_hat" in data:
        vec = data["vec_hat"]
    elif "vec" in data:
        vec = data["vec"]
    else:
        raise KeyError(f"Neither 'vec_hat' nor 'vec' found in {npz_path}")
    return np.asarray(vec, dtype=np.float64)


def fit_lorenz_skeleton_params(vec, X, Y, Z):
    """
    Fixed Lorenz skeleton:
        dx = sigma (y - x)
        dy = rho x - xz - y
        dz = xy - beta z
    Fit only sigma, rho, beta by least squares on the discrete vector field vec[..., 0:3].

    Returns dict with fitted params + residual metrics.
    """
    assert vec.shape[-1] == 3, f"Expected vec[...,3], got shape {vec.shape}"

    dx_obs = vec[..., 0]
    dy_obs = vec[..., 1]
    dz_obs = vec[..., 2]

    # ----- sigma fit from dx = sigma*(Y-X)
    a = (Y - X)
    denom_sigma = np.sum(a * a)
    sigma_hat = np.sum(a * dx_obs) / (denom_sigma + 1e-18)

    # ----- rho fit from dy + xz + y = rho*x
    b = X
    t_rho = dy_obs + X * Z + Y
    denom_rho = np.sum(b * b)
    rho_hat = np.sum(b * t_rho) / (denom_rho + 1e-18)

    # ----- beta fit from xy - dz = beta*z
    c = Z
    t_beta = X * Y - dz_obs
    denom_beta = np.sum(c * c)
    beta_hat = np.sum(c * t_beta) / (denom_beta + 1e-18)

    # Reconstruct Lorenz skeleton vector field with fitted params
    dx_fit = sigma_hat * (Y - X)
    dy_fit = rho_hat * X - X * Z - Y
    dz_fit = X * Y - beta_hat * Z

    resid_x = dx_obs - dx_fit
    resid_y = dy_obs - dy_fit
    resid_z = dz_obs - dz_fit

    mse_x = float(np.mean(resid_x ** 2))
    mse_y = float(np.mean(resid_y ** 2))
    mse_z = float(np.mean(resid_z ** 2))

    resid_sq = resid_x ** 2 + resid_y ** 2 + resid_z ** 2
    obs_sq = dx_obs ** 2 + dy_obs ** 2 + dz_obs ** 2

    mse_total = float(np.mean(resid_sq))
    rmse_total = float(np.sqrt(mse_total))

    obs_energy = float(np.mean(obs_sq))
    rel_rmse = float(np.sqrt(mse_total / (obs_energy + 1e-18)))  # scale-free-ish

    D_hat = float(sigma_hat + 1.0 + beta_hat)      # dissipativity magnitude if >0
    div_hat = float(-(sigma_hat + 1.0 + beta_hat)) # constant divergence in Lorenz skeleton

    return {
        "sigma_hat": float(sigma_hat),
        "rho_hat": float(rho_hat),
        "beta_hat": float(beta_hat),
        "D_hat": D_hat,
        "divergence_hat": div_hat,
        "is_dissipative_hat": bool(D_hat > 0.0),

        "fit_mse_x": mse_x,
        "fit_mse_y": mse_y,
        "fit_mse_z": mse_z,
        "fit_mse_total": mse_total,
        "fit_rmse_total": rmse_total,
        "fit_rel_rmse_total": rel_rmse,
    }


def add_param_errors(fit_result, target_params):
    out = dict(fit_result)
    out["sigma_err_vs_target"] = float(out["sigma_hat"] - target_params["sigma"])
    out["rho_err_vs_target"] = float(out["rho_hat"] - target_params["rho"])
    out["beta_err_vs_target"] = float(out["beta_hat"] - target_params["beta"])
    out["D_target"] = float(target_params["sigma"] + 1.0 + target_params["beta"])
    out["D_err_vs_target"] = float(out["D_hat"] - out["D_target"])
    return out


def summarize_numeric_list(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def group_rows_by_layer_epsilon(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get("kind") != "noisy_reconstruction":
            continue
        key = (int(r["layer_id"]), float(r["epsilon_nominal"]))
        groups[key].append(r)

    summaries = []
    for (layer_id, eps), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        summary = {
            "layer_id": int(layer_id),
            "epsilon_nominal": float(eps),
            "n": len(rs),
            "sigma_hat": summarize_numeric_list([r["sigma_hat"] for r in rs]),
            "rho_hat": summarize_numeric_list([r["rho_hat"] for r in rs]),
            "beta_hat": summarize_numeric_list([r["beta_hat"] for r in rs]),
            "D_hat": summarize_numeric_list([r["D_hat"] for r in rs]),
            "fit_rel_rmse_total": summarize_numeric_list([r["fit_rel_rmse_total"] for r in rs]),
            "reconstruction_mse_vs_input": summarize_numeric_list(
                [r["reconstruction_mse_vs_input"] for r in rs if r.get("reconstruction_mse_vs_input") is not None]
            ),
        }
        summaries.append(summary)
    return summaries


# =========================
# Visualization Helpers
# =========================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _pick_stat_block(block, use_stat="mean"):
    """
    block example:
      {"mean":..., "std":..., "median":..., "q25":..., "q75":..., ...}
    Returns center, low, high for error visualization.
    """
    if block is None:
        return None, None, None

    if use_stat == "median":
        center = block["median"]
        low = block["q25"]
        high = block["q75"]
    else:
        center = block["mean"]
        low = block["mean"] - block["std"]
        high = block["mean"] + block["std"]
    return center, low, high


def build_group_lookup(report):
    """
    Build dict[(layer_id, epsilon)] = group_summary
    """
    lookup = {}
    for g in report["grouped_summary_layer_epsilon"]:
        key = (int(g["layer_id"]), float(g["epsilon_nominal"]))
        lookup[key] = g
    return lookup


def get_layers_epsilons_from_grouped(report):
    groups = report["grouped_summary_layer_epsilon"]
    layers = sorted({int(g["layer_id"]) for g in groups})
    epsilons = sorted({float(g["epsilon_nominal"]) for g in groups})
    return layers, epsilons


def extract_curve_data(report, metric_key, use_stat="mean"):
    """
    metric_key in:
      - "D_hat"
      - "fit_rel_rmse_total"
      - "reconstruction_mse_vs_input"
    Returns:
      layers, epsilons, curves
    where curves[layer_id] = {"x","y","y_low","y_high"}.
    """
    grouped_lookup = build_group_lookup(report)
    layers, epsilons = get_layers_epsilons_from_grouped(report)

    curves = {}
    for layer_id in layers:
        xs, ys, yls, yhs = [], [], [], []
        for eps in epsilons:
            g = grouped_lookup.get((layer_id, eps))
            if g is None:
                continue
            center, low, high = _pick_stat_block(g.get(metric_key), use_stat=use_stat)
            if center is None:
                continue
            xs.append(float(eps))
            ys.append(float(center))
            yls.append(float(low))
            yhs.append(float(high))
        curves[layer_id] = {"x": xs, "y": ys, "y_low": yls, "y_high": yhs}

    return layers, epsilons, curves


def _set_log_x_if_possible(xs_all, enable=True):
    if not enable:
        return
    xs_all = np.asarray(xs_all, dtype=float)
    if xs_all.size == 0:
        return
    if np.all(xs_all > 0):
        plt.xscale("log")


def plot_metric_curves(
    report,
    metric_key,
    ylabel,
    title,
    out_path,
    use_stat="mean",
    x_log_scale=True,
    show_error_band=True,
    transform=None,  # callable on y arrays
):
    layers, _, curves = extract_curve_data(report, metric_key, use_stat=use_stat)

    plt.figure(figsize=(7.2, 4.8), dpi=VIZ_CONFIG["dpi"])

    all_x = []
    for layer_id in layers:
        d = curves[layer_id]
        if len(d["x"]) == 0:
            continue

        x = np.array(d["x"], dtype=float)
        y = np.array(d["y"], dtype=float)
        y_low = np.array(d["y_low"], dtype=float)
        y_high = np.array(d["y_high"], dtype=float)

        if transform is not None:
            y = transform(y)
            y_low = transform(y_low)
            y_high = transform(y_high)

        plt.plot(x, y, marker="o", label=f"Layer {layer_id}")
        if show_error_band:
            lo = np.minimum(y_low, y_high)
            hi = np.maximum(y_low, y_high)
            plt.fill_between(x, lo, hi, alpha=0.2)

        all_x.extend(list(x))

    _set_log_x_if_possible(all_x, enable=x_log_scale)

    plt.xlabel("Noise strength $\\epsilon$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_delta_D_curves(report, out_path, use_stat="mean", x_log_scale=True, show_error_band=True):
    """
    Plot ΔD = D_group - D_clean, one line per layer.
    """
    D_clean = float(report["baseline_clean_reconstruction_fit"]["D_hat"])

    def transform(y):
        return y - D_clean

    stat_name = "mean±std" if use_stat == "mean" else "median+IQR"
    title = f"$\\Delta D$ ({stat_name})"

    plot_metric_curves(
        report=report,
        metric_key="D_hat",
        ylabel="$\\Delta D = D - D_{clean}$",
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=transform,
    )


def plot_fit_rel_rmse_curves(report, out_path, use_stat="mean", x_log_scale=True, show_error_band=True):
    stat_name = "mean±std" if use_stat == "mean" else "median+IQR"
    title = f"Residual error of regression ({stat_name})"
    plot_metric_curves(
        report=report,
        metric_key="fit_rel_rmse_total",
        ylabel="Relative RMSE (Lorenz skeleton fit)",
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=None,
    )


def plot_reconstruction_mse_curves(report, out_path, use_stat="mean", x_log_scale=True, show_error_band=True):
    stat_name = "mean±std" if use_stat == "mean" else "median+IQR"
    title = f"CAE reconstruction MSE ({stat_name})"
    plot_metric_curves(
        report=report,
        metric_key="reconstruction_mse_vs_input",
        ylabel="Reconstruction MSE vs input",
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=None,
    )


def plot_delta_D_heatmap(report, out_path, use_stat="mean"):
    """
    Heatmap over (layer, epsilon) for ΔD = D_group - D_clean.
    Best when more eps levels are available; still fine for MVP.
    """
    D_clean = float(report["baseline_clean_reconstruction_fit"]["D_hat"])
    lookup = build_group_lookup(report)
    layers, epsilons = get_layers_epsilons_from_grouped(report)

    if len(layers) == 0 or len(epsilons) == 0:
        return

    M = np.full((len(layers), len(epsilons)), np.nan, dtype=float)

    for i, layer_id in enumerate(layers):
        for j, eps in enumerate(epsilons):
            g = lookup.get((layer_id, eps))
            if g is None:
                continue
            center, _, _ = _pick_stat_block(g.get("D_hat"), use_stat=use_stat)
            if center is None:
                continue
            M[i, j] = float(center - D_clean)

    plt.figure(figsize=(7.0, 4.6), dpi=VIZ_CONFIG["dpi"])
    im = plt.imshow(M, aspect="equal")
    cbar = plt.colorbar(im)
    cbar.set_label("$\\Delta D = D - D_{clean}$")

    plt.xticks(range(len(epsilons)), [f"{e:g}" for e in epsilons])
    plt.yticks(range(len(layers)), [f"Layer {l}" for l in layers])
    plt.xlabel("Noise strength $\\epsilon$")
    plt.ylabel("Encoder layer")
    plt.title("$\\Delta D$")

    for i in range(len(layers)):
        for j in range(len(epsilons)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", fontsize=5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_mvp_visualizations(report):
    if not VIZ_CONFIG["enabled"]:
        return

    save_dir = VIZ_CONFIG["save_dir"]
    ensure_dir(save_dir)

    use_stat = VIZ_CONFIG["use_stat"]
    x_log_scale = VIZ_CONFIG["x_log_scale"]
    show_error_band = VIZ_CONFIG["show_error_band"]

    plot_delta_D_curves(
        report,
        out_path=os.path.join(save_dir, "curve_deltaD.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )

    plot_fit_rel_rmse_curves(
        report,
        out_path=os.path.join(save_dir, "curve_fit_rel_rmse.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )

    plot_reconstruction_mse_curves(
        report,
        out_path=os.path.join(save_dir, "curve_reconstruction_mse.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )

    if VIZ_CONFIG.get("make_heatmap_deltaD", False):
        plot_delta_D_heatmap(
            report,
            out_path=os.path.join(save_dir, "heatmap_deltaD.png"),
            use_stat=use_stat,
        )

    print(f"\nFigures saved to: {save_dir}")


# =========================
# Main
# =========================
def main():
    run_dir = Path(RUN_DIR)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = load_json(manifest_path)

    model_config = manifest["model_config"]
    grid_shape = model_config["grid_shape"]
    grid_limits = model_config["grid_limits"]

    X, Y, Z = build_xyz_grids(grid_shape, grid_limits)

    # ---- evaluate baseline original + baseline clean reconstruction
    baseline_info = manifest.get("baseline", {})
    original_rel = baseline_info.get("original_vec_file", "baseline/original_vec.npz")
    clean_rel = baseline_info.get("clean_reconstruction_file", "baseline/clean_reconstruction.npz")

    original_path = run_dir / original_rel
    clean_path = run_dir / clean_rel

    original_vec = load_vec_from_npz(original_path)
    clean_vec = load_vec_from_npz(clean_path)

    original_fit = add_param_errors(
        fit_lorenz_skeleton_params(original_vec, X, Y, Z),
        TARGET_LORENZ_PARAMS
    )
    clean_fit = add_param_errors(
        fit_lorenz_skeleton_params(clean_vec, X, Y, Z),
        TARGET_LORENZ_PARAMS
    )

    # ---- evaluate all noisy samples from manifest
    rows = []
    for entry in manifest.get("entries", []):
        if entry.get("kind") != "noisy_reconstruction":
            continue

        rel_file = entry["file"]
        file_path = run_dir / rel_file
        if not file_path.exists():
            print(f"[WARN] Missing file, skip: {file_path}")
            continue

        vec = load_vec_from_npz(file_path)
        fit = add_param_errors(
            fit_lorenz_skeleton_params(vec, X, Y, Z),
            TARGET_LORENZ_PARAMS
        )

        row = {
            "kind": "noisy_reconstruction",
            "file": rel_file,

            "layer_id": int(entry["layer_id"]),
            "epsilon_nominal": float(entry["epsilon_nominal"]),
            "seed_id": int(entry["seed_id"]),

            # from manifest
            "reconstruction_mse_vs_input": entry.get("reconstruction_mse_vs_input"),
            "layer_rms": entry.get("layer_rms"),
            "realized_noise_rms": entry.get("realized_noise_rms"),
            "realized_delta_h_over_h": entry.get("realized_delta_h_over_h"),

            # fitted Lorenz params + residuals + dissipativity
            **fit,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["layer_id"], r["epsilon_nominal"], r["seed_id"]))

    grouped_summary = group_rows_by_layer_epsilon(rows)

    # ---- final report (single file)
    report = {
        "report_type": "LIRT-Lorenz-MVP-evaluation",
        "version": "v1",
        "run_dir": str(run_dir),
        "protocol": manifest.get("protocol"),
        "feature_type": manifest.get("feature_type"),
        "noise_type": manifest.get("noise_type"),
        "target_lorenz_params": TARGET_LORENZ_PARAMS,

        "grid_shape": grid_shape,
        "grid_limits": grid_limits,

        "baseline_original_vector_field_fit": original_fit,
        "baseline_clean_reconstruction_fit": clean_fit,

        "counts": {
            "num_noisy_rows": len(rows),
            "num_group_summaries": len(grouped_summary),
        },

        "rows_noisy": rows,

        "grouped_summary_layer_epsilon": grouped_summary,
    }

    save_json(OUTPUT_JSON, report)

    make_mvp_visualizations(report)

    # ---- console summary
    print("\n===== MVP Evaluation Done =====")
    print(f"Report saved to: {OUTPUT_JSON}")
    print("\n[Original vec fit]")
    print(
        f"sigma={original_fit['sigma_hat']:.6f}, rho={original_fit['rho_hat']:.6f}, beta={original_fit['beta_hat']:.6f}, "
        f"D={original_fit['D_hat']:.6f}, rel_rmse={original_fit['fit_rel_rmse_total']:.6e}"
    )
    print("\n[Clean reconstruction fit]")
    print(
        f"sigma={clean_fit['sigma_hat']:.6f}, rho={clean_fit['rho_hat']:.6f}, beta={clean_fit['beta_hat']:.6f}, "
        f"D={clean_fit['D_hat']:.6f}, rel_rmse={clean_fit['fit_rel_rmse_total']:.6e}"
    )
    print(f"\nNoisy samples evaluated: {len(rows)}")


if __name__ == "__main__":
    main()