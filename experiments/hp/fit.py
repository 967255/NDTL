from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .holling import DEFAULT_DOMAIN, HollingParams, holling_rhs_ecology, unit_to_ecology

PARAM_ORDER = (
    "K",
    "alpha_h",
    "alpha_c",
    "beta_h",
    "beta_c",
    "gamma_h",
    "gamma_c",
    "d_h",
    "d_c",
)


def params_to_array(params: HollingParams):
    return np.array([getattr(params, key) for key in PARAM_ORDER], dtype=np.float64)


def array_to_params(values, name="fit"):
    values = np.asarray(values, dtype=np.float64)
    return HollingParams(
        name=name,
        K=float(values[0]),
        alpha_h=float(values[1]),
        alpha_c=float(values[2]),
        beta_h=float(values[3]),
        beta_c=float(values[4]),
        gamma_h=float(values[5]),
        gamma_c=float(values[6]),
        d_h=float(values[7]),
        d_c=float(values[8]),
    )


def sample_grid_points_and_rhs(
    field_unit, max_samples=12000, domain=DEFAULT_DOMAIN, seed=0
):
    field_unit = np.asarray(field_unit, dtype=np.float64)
    n = field_unit.shape[0]
    axes = [np.linspace(0.0, 1.0, n, dtype=np.float64) for _ in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    unit_points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    rhs_unit = field_unit.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    if unit_points.shape[0] > max_samples:
        idx = rng.choice(unit_points.shape[0], size=int(max_samples), replace=False)
        unit_points = unit_points[idx]
        rhs_unit = rhs_unit[idx]

    eco_points = unit_to_ecology(unit_points, domain=domain)
    spans = np.asarray([b - a for a, b in domain], dtype=np.float64)
    rhs_eco = rhs_unit * spans
    return eco_points, rhs_eco


def _build_profile_system(X: np.ndarray, Y: np.ndarray, gamma_h: float, gamma_c: float):
    p, h, c = X[:, 0], X[:, 1], X[:, 2]
    phi_h = p * h / (gamma_h + p + 1e-12)
    phi_c = h * c / (gamma_c + h + 1e-12)

    n = X.shape[0]
    A = np.zeros((3 * n, 5), dtype=np.float64)
    y = np.zeros(3 * n, dtype=np.float64)

    # theta = [inv_K, q_h, q_c, d_h, d_c], all positive.
    A[:n, 0] = -p * p
    A[:n, 1] = -phi_h
    y[:n] = Y[:, 0] - p

    A[n : 2 * n, 1] = phi_h
    A[n : 2 * n, 2] = -phi_c
    A[n : 2 * n, 3] = -h
    y[n : 2 * n] = Y[:, 1]

    A[2 * n :, 2] = phi_c
    A[2 * n :, 4] = -c
    y[2 * n :] = Y[:, 2]
    return A, y


def _profile_fit_fixed_gamma(
    X: np.ndarray, Y: np.ndarray, gamma_h: float, gamma_c: float
):
    A, y = _build_profile_system(X, Y, gamma_h, gamma_c)
    scale = np.std(Y, axis=0) + 1e-6
    row_scale = np.repeat(scale, X.shape[0])
    A_scaled = A / row_scale[:, None]
    y_scaled = y / row_scale
    result = lsq_linear(
        A_scaled,
        y_scaled,
        bounds=(np.full(5, 1e-12), np.full(5, np.inf)),
        lsmr_tol="auto",
        max_iter=200,
    )
    theta = np.maximum(result.x, 1e-12)
    pred = _predict_profile_rhs(X, gamma_h, gamma_c, theta)
    scaled_resid = np.concatenate(
        [
            (pred[:, 0] - Y[:, 0]) / scale[0],
            (pred[:, 1] - Y[:, 1]) / scale[1],
            (pred[:, 2] - Y[:, 2]) / scale[2],
        ]
    )
    return result, theta, pred, float(np.mean(scaled_resid**2))


def _predict_profile_rhs(
    X: np.ndarray, gamma_h: float, gamma_c: float, theta: np.ndarray
):
    inv_k, q_h, q_c, d_h, d_c = theta
    p, h, c = X[:, 0], X[:, 1], X[:, 2]
    phi_h = p * h / (gamma_h + p + 1e-12)
    phi_c = h * c / (gamma_c + h + 1e-12)
    return np.stack(
        [
            p - inv_k * p * p - q_h * phi_h,
            q_h * phi_h - q_c * phi_c - d_h * h,
            q_c * phi_c - d_c * c,
        ],
        axis=1,
    )


def fit_holling_nonlinear(
    field_unit,
    initial: HollingParams,
    domain=DEFAULT_DOMAIN,
    max_samples: int = 12000,
    max_nfev: int = 250,
    seed: int = 0,
):
    X, Y = sample_grid_points_and_rhs(
        field_unit, max_samples=max_samples, domain=domain, seed=seed
    )
    spans = np.asarray([b - a for a, b in domain], dtype=np.float64)
    lower = np.maximum(1e-4, 0.02 * spans[:2])
    upper = np.maximum(lower * 10.0, 2.5 * spans[:2])
    bounds = np.log(np.stack([lower, upper], axis=1))

    grid_n = 7 if max_nfev <= 60 else 9
    gamma_h_grid = np.geomspace(lower[0], upper[0], grid_n)
    gamma_c_grid = np.geomspace(lower[1], upper[1], grid_n)
    candidates = []
    for gamma_h in gamma_h_grid:
        for gamma_c in gamma_c_grid:
            inner, theta, _, loss = _profile_fit_fixed_gamma(
                X, Y, float(gamma_h), float(gamma_c)
            )
            candidates.append((loss, np.log([gamma_h, gamma_c]), inner, theta))

    init_log = np.log(
        [
            float(np.clip(initial.gamma_h, lower[0], upper[0])),
            float(np.clip(initial.gamma_c, lower[1], upper[1])),
        ]
    )
    inner, theta, _, loss = _profile_fit_fixed_gamma(
        X, Y, float(np.exp(init_log[0])), float(np.exp(init_log[1]))
    )
    candidates.append((loss, init_log, inner, theta))
    candidates.sort(key=lambda item: item[0])

    best = {
        "loss": candidates[0][0],
        "log_gamma": candidates[0][1],
        "inner": candidates[0][2],
        "theta": candidates[0][3],
    }
    nfev = len(candidates)

    def objective(log_gamma):
        nonlocal nfev
        log_gamma = np.asarray(log_gamma, dtype=np.float64)
        gamma_h, gamma_c = np.exp(log_gamma)
        inner_result, theta_result, _, loss_result = _profile_fit_fixed_gamma(
            X, Y, float(gamma_h), float(gamma_c)
        )
        objective.last = (inner_result, theta_result, loss_result)
        nfev += 1
        return loss_result

    objective.last = None
    for _, start, _, _ in candidates[: min(3, len(candidates))]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[tuple(bounds[0]), tuple(bounds[1])],
            options={"maxiter": max(8, int(max_nfev) // 4), "ftol": 1e-10},
        )
        inner_result, theta_result, loss_result = objective.last
        if loss_result < best["loss"]:
            best = {
                "loss": float(loss_result),
                "log_gamma": np.asarray(result.x, dtype=np.float64),
                "inner": inner_result,
                "theta": theta_result,
            }

    gamma_h, gamma_c = np.exp(best["log_gamma"])
    theta = np.asarray(best["theta"], dtype=np.float64)
    inv_k, q_h, q_c, d_h, d_c = theta
    fit_params = HollingParams(
        name="holling_profile_fit",
        K=float(1.0 / inv_k),
        alpha_h=float(q_h),
        alpha_c=float(q_c),
        beta_h=1.0,
        beta_c=1.0,
        gamma_h=float(gamma_h),
        gamma_c=float(gamma_c),
        d_h=float(d_h),
        d_c=float(d_c),
    )
    pred = holling_rhs_ecology(X, fit_params)
    mse = float(np.mean((pred - Y) ** 2))
    rel_rmse = float(np.sqrt(mse / (np.mean(Y**2) + 1e-12)))
    return {
        "success": bool(best["inner"].success),
        "message": "profile gamma search with positive constrained linear inner solve",
        "fit_method": "profile_gamma_positive_lsq",
        "nfev": int(nfev),
        "params": fit_params.to_dict(),
        "effective_params": {
            "inv_K": float(inv_k),
            "q_h": float(q_h),
            "q_c": float(q_c),
            "gamma_h": float(gamma_h),
            "gamma_c": float(gamma_c),
            "d_h": float(d_h),
            "d_c": float(d_c),
        },
        "gamma_bounds": {
            "gamma_h": [float(lower[0]), float(upper[0])],
            "gamma_c": [float(lower[1]), float(upper[1])],
        },
        "profile_scaled_mse": float(best["loss"]),
        "mse": mse,
        "rel_rmse": rel_rmse,
    }


def fit_ecological_basis_linear(
    field_unit,
    gamma_h: float,
    gamma_c: float,
    domain=DEFAULT_DOMAIN,
    max_samples: int = 12000,
    seed: int = 0,
):
    X, Y = sample_grid_points_and_rhs(
        field_unit, max_samples=max_samples, domain=domain, seed=seed
    )
    p, h, c = X[:, 0], X[:, 1], X[:, 2]
    Phi = np.stack(
        [
            np.ones_like(p),
            p,
            p * p,
            h,
            c,
            p * h / (gamma_h + p + 1e-12),
            h * c / (gamma_c + h + 1e-12),
        ],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    pred = Phi @ coef
    mse = float(np.mean((pred - Y) ** 2))
    rel_rmse = float(np.sqrt(mse / (np.mean(Y**2) + 1e-12)))
    return {
        "terms": [
            "1",
            "P",
            "P2",
            "H",
            "C",
            "PH_over_gammaH_plus_P",
            "HC_over_gammaC_plus_H",
        ],
        "coef": coef.T.tolist(),
        "mse": mse,
        "rel_rmse": rel_rmse,
    }
