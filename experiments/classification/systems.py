"""System definitions, numerical integration and distance metrics for classification."""

from __future__ import annotations
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from ndtl.systems import (
    lorenz,
    chen,
    lv,
    rossler,
    burke_shaw,
    dequan_li,
    genesio_tesi,
    hadley,
    halvorsen,
    newton_leipnik,
    nose_hoover,
    rucklidge,
    sakarya,
    shimizu_morioka,
)
from ndtl.metrics import jac_fd_from_ode_d
from ndtl.fields import ODE

TARGET_LIMITS = [[0, 1], [0, 1], [0, 1]]

TARGET_VELOCITY = 5.0

DEFAULT_ALPHAS = np.linspace(0.0, 1.0, 9)

DEFAULT_INIT_POINT = (0.3, 0.3, 0.5)

STANDARD_TERMS_3D_QUAD = ["", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]


@dataclass(frozen=True)
class SystemSpec:
    key: str
    label: str
    source: str
    ode: ODE
    x0: tuple[float, float, float]
    bbox_h: float
    bbox_burn: int = 5000
    bbox_steps: int = 25000
    bbox_max_abs: float = 1e5


@dataclass(frozen=True)
class LyapConfig:
    h: float = 0.01
    burn_steps: int = 1000
    lyap_steps: int = 5000
    fd_eps: float | None = None
    max_abs: float = 1e5


def clone_ode(
    ode: ODE, name: str | None = None, grid_limits: list[list[float]] | None = None
) -> ODE:
    limits = grid_limits
    if limits is None and getattr(ode, "grid_limits", None) is not None:
        limits = [[float(a), float(b)] for a, b in ode.grid_limits]

    return ODE(
        name=name if name is not None else ode.name,
        dimension=ode.dimension,
        max_order=ode.max_order,
        coef=np.array(ode.coef, dtype=float, copy=True),
        h=float(getattr(ode, "h", 0.01)),
        extra_term_dicts=dict(getattr(ode, "extra_term_dicts", {})),
        grid_limits=limits,
        system_velocity=getattr(ode, "system_velocity", None),
    )


def make_dadras() -> ODE:
    ode = ODE(name="dadras", dimension=3, max_order=2)
    ode.set_coef(0, "y", 1.0)
    ode.set_coef(0, "x", -3.0)
    ode.set_coef(0, "yz", 2.7)
    ode.set_coef(1, "y", 1.7)
    ode.set_coef(1, "xz", -1.0)
    ode.set_coef(1, "z", 1.0)
    ode.set_coef(2, "xy", 2.0)
    ode.set_coef(2, "z", -9.0)
    ode.h = 0.003
    return ode


def make_rikitake_dynamo() -> ODE:
    ode = ODE(name="rikitake_dynamo", dimension=3, max_order=2)
    ode.set_coef(0, "x", -1.0)
    ode.set_coef(0, "yz", 1.0)
    ode.set_coef(1, "x", -1.0)
    ode.set_coef(1, "y", -1.0)
    ode.set_coef(1, "xz", 1.0)
    ode.set_coef(2, "", 1.0)
    ode.set_coef(2, "xy", -1.0)
    ode.h = 0.005
    return ode


def make_pehlivan_wei() -> ODE:
    ode = ODE(name="pehlivan_wei", dimension=3, max_order=2)
    ode.set_coef(0, "y", 1.0)
    ode.set_coef(0, "yz", -1.0)
    ode.set_coef(1, "y", 1.0)
    ode.set_coef(1, "yz", 1.0)
    ode.set_coef(1, "x", -2.0)
    ode.set_coef(2, "", 2.0)
    ode.set_coef(2, "xy", -1.0)
    ode.set_coef(2, "yy", -1.0)
    ode.h = 0.005
    return ode


def make_chen_lee() -> ODE:
    ode = ODE(name="chen_lee", dimension=3, max_order=2)
    ode.set_coef(0, "x", 5.0)
    ode.set_coef(0, "yz", -1.0)
    ode.set_coef(1, "y", -10.0)
    ode.set_coef(1, "xz", 1.0)
    ode.set_coef(2, "z", -0.38)
    ode.set_coef(2, "xy", 1.0 / 3.0)
    ode.h = 0.003
    return ode


def make_finance() -> ODE:
    ode = ODE(name="finance", dimension=3, max_order=2)
    ode.set_coef(0, "x", 1.0 / 0.2 - 0.001)
    ode.set_coef(0, "z", 1.0)
    ode.set_coef(0, "xy", 1.0)
    ode.set_coef(1, "y", -0.2)
    ode.set_coef(1, "xx", -1.0)
    ode.set_coef(2, "x", -1.0)
    ode.set_coef(2, "z", -1.1)
    ode.h = 0.003
    return ode


def make_sprott_b() -> ODE:
    ode = ODE(name="sprott_b", dimension=3, max_order=2)
    ode.set_coef(0, "yz", 1.0)
    ode.set_coef(1, "x", 1.0)
    ode.set_coef(1, "y", -1.0)
    ode.set_coef(2, "", 1.0)
    ode.set_coef(2, "xy", -1.0)
    ode.h = 0.01
    return ode


def build_system_specs() -> list[SystemSpec]:
    return [
        SystemSpec(
            "lorenz", "Lorenz", "existing", clone_ode(lorenz), (1.0, 1.0, 1.0), 0.005
        ),
        SystemSpec(
            "chen",
            "Chen",
            "existing",
            clone_ode(chen),
            (3.2689697, 5.2990458, 19.435349),
            0.0015,
        ),
        SystemSpec("lv", "Lv", "existing", clone_ode(lv), (1.0, 1.0, 1.0), 0.0015),
        SystemSpec(
            "rossler",
            "Rossler",
            "existing",
            clone_ode(rossler),
            (6.5134412, 0.4772013, 0.34164294),
            0.01,
        ),
        SystemSpec(
            "burke_shaw",
            "Burke-Shaw",
            "existing",
            clone_ode(burke_shaw),
            (-0.79153188, 0.65244479, 0.22517831),
            0.0015,
        ),
        SystemSpec(
            "dequan_li",
            "Dequan Li",
            "existing",
            clone_ode(dequan_li),
            (-11.395722, -104.12614, 202.46173),
            0.0002,
            bbox_burn=8000,
            bbox_steps=30000,
            bbox_max_abs=1e6,
        ),
        SystemSpec(
            "genesio_tesi",
            "Genesio-Tesi",
            "existing",
            clone_ode(genesio_tesi),
            (-0.053772565, -0.26162564, 0.33501557),
            0.01,
        ),
        SystemSpec(
            "hadley", "Hadley", "existing", clone_ode(hadley), (0.1, 0.0, 0.0), 0.005
        ),
        SystemSpec(
            "halvorsen",
            "Halvorsen",
            "existing",
            clone_ode(halvorsen),
            (-8.6807408, -2.4741399, 0.070775762),
            0.0015,
            bbox_max_abs=1e6,
        ),
        SystemSpec(
            "newton_leipnik",
            "Newton-Leipnik",
            "existing",
            clone_ode(newton_leipnik),
            (0.29430753, 0.10649486, 0.29081125),
            0.005,
        ),
        SystemSpec(
            "nose_hoover",
            "Nose-Hoover",
            "existing",
            clone_ode(nose_hoover),
            (-1.9758582, 0.24647527, 0.79024299),
            0.01,
        ),
        SystemSpec(
            "rucklidge",
            "Rucklidge",
            "existing",
            clone_ode(rucklidge),
            (1.9257158, 3.1923835, 5.7829655),
            0.005,
        ),
        SystemSpec(
            "sakarya",
            "Sakarya",
            "existing",
            clone_ode(sakarya),
            (-2.8976045, 3.8877978, 3.07465),
            0.003,
            bbox_max_abs=1e6,
        ),
        SystemSpec(
            "shimizu_morioka",
            "Shimizu-Morioka",
            "existing",
            clone_ode(shimizu_morioka),
            (0.016838697, 0.03489384, 0.35346848),
            0.01,
        ),
        SystemSpec(
            "dadras",
            "Dadras",
            "DYSTS",
            make_dadras(),
            (1.0284085, -3.5348655, -0.72081646),
            0.003,
        ),
        SystemSpec(
            "rikitake_dynamo",
            "Rikitake Dynamo",
            "DYSTS",
            make_rikitake_dynamo(),
            (-0.7791562791136255, -0.6302244598741468, 2.671200187214492),
            0.005,
        ),
        SystemSpec(
            "pehlivan_wei",
            "Pehlivan-Wei",
            "DYSTS",
            make_pehlivan_wei(),
            (0.21665608, 0.86710502, 0.50627859),
            0.005,
        ),
        SystemSpec(
            "chen_lee",
            "Chen-Lee",
            "DYSTS",
            make_chen_lee(),
            (0.17177112, 0.10360479, 6.2553878),
            0.003,
        ),
        SystemSpec(
            "finance",
            "Finance",
            "DYSTS",
            make_finance(),
            (-0.51664739, -3.9568707, 0.0012893581),
            0.003,
            bbox_max_abs=1e6,
        ),
        SystemSpec(
            "sprott_b",
            "Sprott B",
            "DYSTS",
            make_sprott_b(),
            (1.6878282, 1.4848132, -0.53706628),
            0.01,
        ),
    ]


def has_standard_quadratic_basis(ode: ODE) -> bool:
    return (
        int(ode.dimension) == 3
        and int(ode.max_order) == 2
        and list(getattr(ode, "terms", [])) == STANDARD_TERMS_3D_QUAD
        and not getattr(ode, "extra_term_dicts", {})
    )


def quadratic_basis3(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = np.asarray(x, dtype=float).reshape(3)
    return np.array(
        [1.0, x0, x1, x2, x0 * x0, x0 * x1, x0 * x2, x1 * x1, x1 * x2, x2 * x2],
        dtype=float,
    )


def rhs_fast(ode: ODE, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if has_standard_quadratic_basis(ode):
        return quadratic_basis3(x) @ np.asarray(ode.coef, dtype=float).T
    return np.asarray(ode.d(x.reshape(1, -1)), dtype=float).reshape(-1)


def rk4_step_fast(ode: ODE, x: np.ndarray, h: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    k1 = rhs_fast(ode, x)
    k2 = rhs_fast(ode, x + 0.5 * h * k1)
    k3 = rhs_fast(ode, x + 0.5 * h * k2)
    k4 = rhs_fast(ode, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def jac_fd_fast(ode: ODE, x: np.ndarray, eps: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if not has_standard_quadratic_basis(ode):
        return jac_fd_from_ode_d(ode, x, eps=eps)

    d = x.size
    if eps is None:
        eps = 1e-6 * (1.0 + np.linalg.norm(x))

    J = np.zeros((d, d), dtype=float)
    for j in range(d):
        dx = np.zeros(d, dtype=float)
        dx[j] = eps
        fp = rhs_fast(ode, x + dx)
        fm = rhs_fast(ode, x - dx)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def integrate_one_step(ode: ODE, x: np.ndarray, h: float, max_abs: float) -> np.ndarray:
    x_next = rk4_step_fast(ode, x.reshape(-1), h=h).reshape(-1)
    if (not np.all(np.isfinite(x_next))) or np.max(np.abs(x_next)) > max_abs:
        raise FloatingPointError("trajectory exploded or became non-finite")
    return x_next


def compute_lyapunov_light(
    ode: ODE,
    init_point: tuple[float, float, float],
    cfg: LyapConfig,
) -> dict[str, Any]:
    x = np.asarray(init_point, dtype=float).reshape(-1)
    d = int(ode.dimension)
    if x.size != d:
        raise ValueError(f"init dimension mismatch: {x.size} vs {d}")

    for _ in range(cfg.burn_steps):
        x = integrate_one_step(ode, x, cfg.h, cfg.max_abs)

    U = np.eye(d, dtype=float)
    logs: list[np.ndarray] = []

    for _ in range(cfg.lyap_steps):
        x = integrate_one_step(ode, x, cfg.h, cfg.max_abs)
        J = jac_fd_fast(ode, x, eps=cfg.fd_eps)
        A = np.eye(d, dtype=float) - J * cfg.h
        U_new = np.linalg.solve(A, U)
        Q, R = np.linalg.qr(U_new)
        logs.append(np.log(np.abs(np.diag(R)) + 1e-300))
        U = Q

    logs_arr = np.asarray(logs, dtype=float)
    spectrum = np.sum(logs_arr, axis=0) / (cfg.h * cfg.lyap_steps)
    spectrum = np.sort(spectrum)[::-1]
    return {
        "spectrum": spectrum,
        "dt": cfg.h,
        "used_steps": cfg.lyap_steps,
    }


def format_csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.12g}"
    return value


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "system_a",
        "system_b",
        "alpha_index",
        "alpha",
        "status",
        "lambda1",
        "lambda2",
        "lambda3",
        "used_steps",
        "dt",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: format_csv_value(row.get(k, "")) for k in fieldnames})


def spectra_from_records(
    records: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    by_pair = {
        f"{a}__{b}": np.full((len(alphas), 3), np.nan, dtype=float) for a, b in pairs
    }
    for row in records:
        if row["status"] != "OK":
            continue
        pair_id = str(row["pair_id"])
        i = int(row["alpha_index"])
        by_pair[pair_id][i, :] = [
            float(row["lambda1"]),
            float(row["lambda2"]),
            float(row["lambda3"]),
        ]
    return by_pair


def pair_components(
    pair_id: str, spectra: np.ndarray, alphas: np.ndarray
) -> dict[str, float]:
    valid = np.all(np.isfinite(spectra), axis=1)
    chaos_bad = (~valid).astype(float)
    chaos_bad[valid] = (spectra[valid, 0] <= 0.0).astype(float)
    chaos_loss = float(np.mean(chaos_bad))

    endpoint_gap = np.nan
    barrier = np.nan
    roughness = np.nan

    if valid[0] and valid[-1]:
        spec0 = spectra[0]
        spec1 = spectra[-1]
        endpoint_gap = float(np.linalg.norm(spec1 - spec0))

        vals = []
        for i in range(1, len(alphas) - 1):
            if valid[i]:
                expected = (1.0 - alphas[i]) * spec0 + alphas[i] * spec1
                vals.append(float(np.linalg.norm(spectra[i] - expected)))
        if vals:
            barrier = float(np.mean(vals))

    vals = []
    for i in range(1, len(alphas) - 1):
        if valid[i - 1] and valid[i] and valid[i + 1]:
            vals.append(
                float(
                    np.linalg.norm(spectra[i + 1] - 2.0 * spectra[i] + spectra[i - 1])
                )
            )
    if vals:
        roughness = float(np.mean(vals))

    return {
        "pair_id": pair_id,
        "endpoint_gap": endpoint_gap,
        "barrier": barrier,
        "roughness": roughness,
        "chaos_loss": chaos_loss,
    }


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    work = values.copy()
    finite = np.isfinite(work)
    if np.any(finite):
        work[~finite] = np.nanmax(work[finite]) + 1.0
    else:
        work[:] = 1.0

    n = work.size
    if n <= 1:
        return np.zeros_like(work)

    order = np.argsort(work, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and work[order[end]] == work[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks / float(n - 1)


def build_pair_component_rows(
    pairs: list[tuple[str, str]],
    alphas: np.ndarray,
    spectra_by_pair: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name_a, name_b in pairs:
        pair_id = f"{name_a}__{name_b}"
        row = {
            "system_a": name_a,
            "system_b": name_b,
            **pair_components(pair_id, spectra_by_pair[pair_id], alphas),
        }
        rows.append(row)

    component_names = ["endpoint_gap", "barrier", "roughness", "chaos_loss"]
    for component in component_names:
        ranks = percentile_ranks(
            np.array([float(row[component]) for row in rows], dtype=float)
        )
        for row, rank in zip(rows, ranks):
            row[f"{component}_rank"] = float(rank)

    for row in rows:
        row["distance"] = float(np.mean([row[f"{c}_rank"] for c in component_names]))

    return rows


def write_pair_components(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "system_a",
        "system_b",
        "endpoint_gap",
        "barrier",
        "roughness",
        "chaos_loss",
        "endpoint_gap_rank",
        "barrier_rank",
        "roughness_rank",
        "chaos_loss_rank",
        "distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_csv_value(row.get(k, "")) for k in fieldnames})


def matrix_from_pair_rows(
    names: list[str],
    rows: list[dict[str, Any]],
    value_name: str,
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for row in rows:
        i = idx[row["system_a"]]
        j = idx[row["system_b"]]
        value = float(row[value_name])
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[format_csv_value(float(v)) for v in row]])


def write_nearest_neighbors(
    path: Path, names: list[str], distance_matrix: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "rank", "neighbor", "distance"]
        )
        writer.writeheader()
        for i, name in enumerate(names):
            order = np.argsort(distance_matrix[i], kind="mergesort")
            rank = 1
            for j in order:
                if i == j:
                    continue
                writer.writerow(
                    {
                        "system": name,
                        "rank": rank,
                        "neighbor": names[j],
                        "distance": format_csv_value(float(distance_matrix[i, j])),
                    }
                )
                rank += 1


def parse_alpha_list(text: str | None) -> np.ndarray:
    if text is None:
        return DEFAULT_ALPHAS.copy()
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError("--alphas must contain at least two values")
    alphas = np.asarray(values, dtype=float)
    if np.any(alphas < 0.0) or np.any(alphas > 1.0):
        raise ValueError("all alpha values must be in [0, 1]")
    return alphas
