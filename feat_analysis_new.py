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

# Visualization
VIZ_CONFIG = {
    "enabled": True,
    "use_stat": "mean",            # "mean" or "median"
    "x_log_scale": True,           # future-proof for log-spaced eps
    "show_error_band": True,       # std band if mean, IQR band if median
    "dpi": 220,
    "save_dir": os.path.join(RUN_DIR, "figures"),
    "make_heatmap_deltaD": True,
    "make_heatmap_lambda1": True,
    "make_heatmap_ky": True,
}

# Lyapunov / KY for fitted Lorenz systems
LYAP_CONFIG = {
    "enabled": True,
    "initial_state": [1.0, 1.0, 1.0],
    "dt": 0.01,
    "transient_steps": 2000,
    "qr_steps": 5,                      # QR re-orthonormalization interval
    "n_intervals": 2000,                # total measured steps = qr_steps * n_intervals
    "divergence_threshold": 1e6,
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
    arr = arr[np.isfinite(arr)]
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


# =========================
# Lyapunov / KY Helpers
# =========================
def lorenz_rhs(x, sigma, rho, beta):
    return np.array([
        sigma * (x[1] - x[0]),
        rho * x[0] - x[0] * x[2] - x[1],
        x[0] * x[1] - beta * x[2],
    ], dtype=np.float64)


def lorenz_jacobian(x, sigma, rho, beta):
    return np.array([
        [-sigma,      sigma,      0.0],
        [rho - x[2],  -1.0,      -x[0]],
        [x[1],        x[0],      -beta],
    ], dtype=np.float64)


def combined_rhs_lorenz(y, sigma, rho, beta):
    """
    y = [x(3), Phi(9)] where dPhi/dt = J(x) Phi
    """
    x = y[:3]
    Phi = y[3:].reshape(3, 3)
    dx = lorenz_rhs(x, sigma, rho, beta)
    J = lorenz_jacobian(x, sigma, rho, beta)
    dPhi = J @ Phi
    return np.concatenate([dx, dPhi.reshape(-1)])


def rk4_step_state(x, dt, sigma, rho, beta):
    k1 = lorenz_rhs(x, sigma, rho, beta)
    k2 = lorenz_rhs(x + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_rhs(x + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_rhs(x + dt * k3, sigma, rho, beta)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_step_combined(y, dt, sigma, rho, beta):
    k1 = combined_rhs_lorenz(y, sigma, rho, beta)
    k2 = combined_rhs_lorenz(y + 0.5 * dt * k1, sigma, rho, beta)
    k3 = combined_rhs_lorenz(y + 0.5 * dt * k2, sigma, rho, beta)
    k4 = combined_rhs_lorenz(y + dt * k3, sigma, rho, beta)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def kaplan_yorke_dimension(lambdas):
    lambdas = np.sort(np.asarray(lambdas, dtype=np.float64))[::-1]
    if not np.all(np.isfinite(lambdas)):
        return np.nan

    csum = np.cumsum(lambdas)

    if csum[0] < 0:
        return 0.0
    if csum[1] < 0:
        return float(1.0 + lambdas[0] / (abs(lambdas[1]) + 1e-18))
    if csum[2] < 0:
        return float(2.0 + (lambdas[0] + lambdas[1]) / (abs(lambdas[2]) + 1e-18))
    return 3.0


def compute_lorenz_lyapunov_spectrum(
    sigma,
    rho,
    beta,
    x0=None,
    dt=0.01,
    transient_steps=2000,
    qr_steps=5,
    n_intervals=2000,
    divergence_threshold=1e6,
):
    """
    Benettin QR method for the fitted Lorenz system.
    Returns:
        {
          "lambda1_hat", "lambda2_hat", "lambda3_hat", "ky_dim_hat",
          "lyap_success", "lyap_message"
        }
    """
    if x0 is None:
        x0 = LYAP_CONFIG["initial_state"]

    try:
        x = np.asarray(x0, dtype=np.float64).copy()

        # transient
        for _ in range(int(transient_steps)):
            x = rk4_step_state(x, dt, sigma, rho, beta)
            if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > divergence_threshold):
                return {
                    "lambda1_hat": np.nan,
                    "lambda2_hat": np.nan,
                    "lambda3_hat": np.nan,
                    "ky_dim_hat": np.nan,
                    "lyap_success": False,
                    "lyap_message": "Diverged during transient",
                }

        Q = np.eye(3, dtype=np.float64)
        sums = np.zeros(3, dtype=np.float64)

        for _ in range(int(n_intervals)):
            y = np.concatenate([x, Q.reshape(-1)])

            for _ in range(int(qr_steps)):
                y = rk4_step_combined(y, dt, sigma, rho, beta)
                x_now = y[:3]
                if (not np.all(np.isfinite(y))) or (np.linalg.norm(x_now) > divergence_threshold):
                    return {
                        "lambda1_hat": np.nan,
                        "lambda2_hat": np.nan,
                        "lambda3_hat": np.nan,
                        "ky_dim_hat": np.nan,
                        "lyap_success": False,
                        "lyap_message": "Diverged during spectrum integration",
                    }

            x = y[:3]
            Phi = y[3:].reshape(3, 3)

            Q, R = np.linalg.qr(Phi)
            diagR = np.diag(R).copy()

            # stabilize sign so log(abs(diagR)) is well-behaved
            signs = np.sign(diagR)
            signs[signs == 0] = 1.0
            S = np.diag(signs)
            Q = Q @ S
            diagR = diagR * signs

            sums += np.log(np.abs(diagR) + 1e-30)

        total_time = float(dt * qr_steps * n_intervals)
        lambdas = sums / total_time
        lambdas = np.sort(lambdas)[::-1]
        ky_dim = kaplan_yorke_dimension(lambdas)

        return {
            "lambda1_hat": float(lambdas[0]),
            "lambda2_hat": float(lambdas[1]),
            "lambda3_hat": float(lambdas[2]),
            "ky_dim_hat": float(ky_dim),
            "lyap_success": True,
            "lyap_message": "OK",
        }

    except Exception as e:
        return {
            "lambda1_hat": np.nan,
            "lambda2_hat": np.nan,
            "lambda3_hat": np.nan,
            "ky_dim_hat": np.nan,
            "lyap_success": False,
            "lyap_message": f"Exception: {repr(e)}",
        }


def add_lyap_metrics(fit_result):
    """
    Compute Lyapunov spectrum / KY for the fitted Lorenz system.
    """
    if not LYAP_CONFIG["enabled"]:
        out = dict(fit_result)
        out["lambda1_hat"] = np.nan
        out["lambda2_hat"] = np.nan
        out["lambda3_hat"] = np.nan
        out["ky_dim_hat"] = np.nan
        out["lyap_success"] = False
        out["lyap_message"] = "LYAP_CONFIG['enabled']=False"
        return out

    lyap = compute_lorenz_lyapunov_spectrum(
        sigma=fit_result["sigma_hat"],
        rho=fit_result["rho_hat"],
        beta=fit_result["beta_hat"],
        x0=LYAP_CONFIG["initial_state"],
        dt=LYAP_CONFIG["dt"],
        transient_steps=LYAP_CONFIG["transient_steps"],
        qr_steps=LYAP_CONFIG["qr_steps"],
        n_intervals=LYAP_CONFIG["n_intervals"],
        divergence_threshold=LYAP_CONFIG["divergence_threshold"],
    )
    out = dict(fit_result)
    out.update(lyap)
    return out


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
            "lyap_success_n": int(sum(bool(r.get("lyap_success", False)) for r in rs)),

            "sigma_hat": summarize_numeric_list([r["sigma_hat"] for r in rs]),
            "rho_hat": summarize_numeric_list([r["rho_hat"] for r in rs]),
            "beta_hat": summarize_numeric_list([r["beta_hat"] for r in rs]),
            "D_hat": summarize_numeric_list([r["D_hat"] for r in rs]),

            "lambda1_hat": summarize_numeric_list([r["lambda1_hat"] for r in rs]),
            "lambda2_hat": summarize_numeric_list([r["lambda2_hat"] for r in rs]),
            "lambda3_hat": summarize_numeric_list([r["lambda3_hat"] for r in rs]),
            "ky_dim_hat": summarize_numeric_list([r["ky_dim_hat"] for r in rs]),

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
      - "lambda1_hat", "lambda2_hat", "lambda3_hat"
      - "ky_dim_hat"
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

    plt.xlabel("Noise strength $\\epsilon$", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # plt.title(title)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_heatmap(
    report,
    metric_key,
    colorbar_label,
    title,
    out_path,
    use_stat="mean",
    transform=None,
    annotate=True,
    annotate_fmt=".3g",
):
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
            center, _, _ = _pick_stat_block(g.get(metric_key), use_stat=use_stat)
            if center is None:
                continue
            val = float(center)
            if transform is not None:
                val = float(transform(np.array([val]))[0])
            M[i, j] = val

    plt.figure(figsize=(7.0, 4.6), dpi=VIZ_CONFIG["dpi"])
    im = plt.imshow(M, aspect="equal")
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)

    plt.xticks(range(len(epsilons)), [f"{e:g}" for e in epsilons])
    plt.yticks(range(len(layers)), [f"Layer {l}" for l in layers])
    plt.xlabel("Noise strength $\\epsilon$")
    plt.ylabel("Encoder layer")
    plt.title(title)

    if annotate:
        for i in range(len(layers)):
            for j in range(len(epsilons)):
                if np.isfinite(M[i, j]):
                    plt.text(j, i, format(M[i, j], annotate_fmt), ha="center", va="center", fontsize=5)

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
        # ylabel="$\\Delta D = D - D_{clean}$",
        ylabel="$\\Delta D$",
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
        # ylabel="Relative RMSE (Lorenz skeleton fit)",
        ylabel="Relative RMSE",
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
        # ylabel="Reconstruction MSE vs input",
        ylabel="Reconstruction MSE",
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=None,
    )


def plot_delta_D_heatmap(report, out_path, use_stat="mean"):
    D_clean = float(report["baseline_clean_reconstruction_fit"]["D_hat"])

    def transform(y):
        return y - D_clean

    plot_metric_heatmap(
        report=report,
        metric_key="D_hat",
        colorbar_label="$\\Delta D = D - D_{clean}$",
        title="$\\Delta D$",
        out_path=out_path,
        use_stat=use_stat,
        transform=transform,
        annotate=True,
        annotate_fmt=".3g",
    )


def plot_lambda_curves(report, metric_key, out_path, use_stat="mean", x_log_scale=True, show_error_band=True):
    pretty = {
        "lambda1_hat": "$\\lambda_1$",
        "lambda2_hat": "$\\lambda_2$",
        "lambda3_hat": "$\\lambda_3$",
    }[metric_key]
    stat_name = "mean±std" if use_stat == "mean" else "median+IQR"
    title = f"{pretty} ({stat_name})"
    plot_metric_curves(
        report=report,
        metric_key=metric_key,
        ylabel=pretty,
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=None,
    )


def plot_ky_curves(report, out_path, use_stat="mean", x_log_scale=True, show_error_band=True):
    stat_name = "mean±std" if use_stat == "mean" else "median+IQR"
    title = f"Kaplan--Yorke dimension ({stat_name})"
    plot_metric_curves(
        report=report,
        metric_key="ky_dim_hat",
        # ylabel="$D_{KY}$",
        ylabel="Kaplan-Yorke dimension",
        title=title,
        out_path=out_path,
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
        transform=None,
    )


def plot_lambda1_heatmap(report, out_path, use_stat="mean"):
    plot_metric_heatmap(
        report=report,
        metric_key="lambda1_hat",
        colorbar_label="$\\lambda_1$",
        title="$\\lambda_1$",
        out_path=out_path,
        use_stat=use_stat,
        transform=None,
        annotate=True,
        annotate_fmt=".3g",
    )


def plot_ky_heatmap(report, out_path, use_stat="mean"):
    plot_metric_heatmap(
        report=report,
        metric_key="ky_dim_hat",
        colorbar_label="$D_{KY}$",
        title="$D_{KY}$",
        out_path=out_path,
        use_stat=use_stat,
        transform=None,
        annotate=True,
        annotate_fmt=".3g",
    )


def make_visualizations(report):
    if not VIZ_CONFIG["enabled"]:
        return

    save_dir = VIZ_CONFIG["save_dir"]
    ensure_dir(save_dir)

    use_stat = VIZ_CONFIG["use_stat"]
    x_log_scale = VIZ_CONFIG["x_log_scale"]
    show_error_band = VIZ_CONFIG["show_error_band"]

    # original experiments
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

    # new experiments: fitted Lorenz Lyapunov spectrum / KY
    plot_lambda_curves(
        report,
        metric_key="lambda1_hat",
        out_path=os.path.join(save_dir, "curve_lambda1.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )
    plot_lambda_curves(
        report,
        metric_key="lambda2_hat",
        out_path=os.path.join(save_dir, "curve_lambda2.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )
    plot_lambda_curves(
        report,
        metric_key="lambda3_hat",
        out_path=os.path.join(save_dir, "curve_lambda3.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )
    plot_ky_curves(
        report,
        out_path=os.path.join(save_dir, "curve_ky_dim.png"),
        use_stat=use_stat,
        x_log_scale=x_log_scale,
        show_error_band=show_error_band,
    )

    if VIZ_CONFIG.get("make_heatmap_lambda1", False):
        plot_lambda1_heatmap(
            report,
            out_path=os.path.join(save_dir, "heatmap_lambda1.png"),
            use_stat=use_stat,
        )

    if VIZ_CONFIG.get("make_heatmap_ky", False):
        plot_ky_heatmap(
            report,
            out_path=os.path.join(save_dir, "heatmap_ky_dim.png"),
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
    original_fit = add_lyap_metrics(original_fit)

    clean_fit = add_param_errors(
        fit_lorenz_skeleton_params(clean_vec, X, Y, Z),
        TARGET_LORENZ_PARAMS
    )
    clean_fit = add_lyap_metrics(clean_fit)

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
        fit = add_lyap_metrics(fit)

        row = {
            "kind": "noisy_reconstruction",
            "file": rel_file,

            "layer_id": int(entry["layer_id"]),
            "epsilon_nominal": float(entry["epsilon_nominal"]),
            "seed_id": int(entry["seed_id"]),

            # from manifest (if present)
            "reconstruction_mse_vs_input": entry.get("reconstruction_mse_vs_input"),
            "layer_rms": entry.get("layer_rms"),
            "realized_noise_rms": entry.get("realized_noise_rms"),
            "realized_delta_h_over_h": entry.get("realized_delta_h_over_h"),

            # fitted Lorenz params + residuals + dissipativity + lyap / ky
            **fit,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["layer_id"], r["epsilon_nominal"], r["seed_id"]))

    grouped_summary = group_rows_by_layer_epsilon(rows)

    # ---- final report (single file)
    report = {
        "report_type": "LIRT-Lorenz-evaluation",
        "version": "v2_with_lyap_ky",
        "run_dir": str(run_dir),
        "protocol": manifest.get("protocol"),
        "feature_type": manifest.get("feature_type"),
        "noise_type": manifest.get("noise_type"),
        "target_lorenz_params": TARGET_LORENZ_PARAMS,
        "lyap_config": LYAP_CONFIG,

        "grid_shape": grid_shape,
        "grid_limits": grid_limits,

        "baseline_original_vector_field_fit": original_fit,
        "baseline_clean_reconstruction_fit": clean_fit,

        "counts": {
            "num_noisy_rows": len(rows),
            "num_group_summaries": len(grouped_summary),
            "num_lyap_success_rows": int(sum(bool(r.get("lyap_success", False)) for r in rows)),
        },

        "rows_noisy": rows,

        "grouped_summary_layer_epsilon": grouped_summary,
    }

    save_json(OUTPUT_JSON, report)

    # ---- visualization
    make_visualizations(report)

    # ---- console summary
    print("\n===== Evaluation Done =====")
    print(f"Report saved to: {OUTPUT_JSON}")

    print("\n[Original vec fit]")
    print(
        f"sigma={original_fit['sigma_hat']:.6f}, rho={original_fit['rho_hat']:.6f}, beta={original_fit['beta_hat']:.6f}, "
        f"D={original_fit['D_hat']:.6f}, rel_rmse={original_fit['fit_rel_rmse_total']:.6e}, "
        f"lambda=({original_fit['lambda1_hat']:.6f}, {original_fit['lambda2_hat']:.6f}, {original_fit['lambda3_hat']:.6f}), "
        f"KY={original_fit['ky_dim_hat']:.6f}"
    )

    print("\n[Clean reconstruction fit]")
    print(
        f"sigma={clean_fit['sigma_hat']:.6f}, rho={clean_fit['rho_hat']:.6f}, beta={clean_fit['beta_hat']:.6f}, "
        f"D={clean_fit['D_hat']:.6f}, rel_rmse={clean_fit['fit_rel_rmse_total']:.6e}, "
        f"lambda=({clean_fit['lambda1_hat']:.6f}, {clean_fit['lambda2_hat']:.6f}, {clean_fit['lambda3_hat']:.6f}), "
        f"KY={clean_fit['ky_dim_hat']:.6f}"
    )

    print(f"\nNoisy samples evaluated: {len(rows)}")
    print(f"Lyapunov-spectrum successes: {report['counts']['num_lyap_success_rows']} / {len(rows)}")


if __name__ == "__main__":
    main()