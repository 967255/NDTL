from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np

DOMAIN_PRESETS = {
    "fixed15": ((0.0, 1.5), (0.0, 1.5), (0.0, 15.0)),
    "fixed02": ((0.0, 2.0), (0.0, 2.0), (0.0, 15.0)),
}

DEFAULT_DOMAIN = DOMAIN_PRESETS["fixed02"]


@dataclass(frozen=True)
class HollingParams:
    name: str
    K: float = 1.0
    alpha_h: float = 0.4
    alpha_c: float = 0.08
    beta_h: float = 2.009
    beta_c: float = 2.876
    gamma_h: float = 0.16129
    gamma_c: float = 0.5
    d_h: float = 0.4
    d_c: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)


def hp_dimensionless_params(name: str, b1: float) -> HollingParams:
    """Represent the original Hastings-Powell dimensionless family."""

    a1 = 5.0
    a2 = 0.1
    b2 = 2.0
    return HollingParams(
        name=name,
        K=1.0,
        alpha_h=a1 / b1,
        beta_h=1.0,
        gamma_h=1.0 / b1,
        alpha_c=a2 / b2,
        beta_c=1.0,
        gamma_c=1.0 / b2,
        d_h=0.4,
        d_c=0.01,
    )


HP_FALLBACK_MODULES = {
    "stable": hp_dimensionless_params("hp_b1_2p0_stable", 2.0),
    "periodic": hp_dimensionless_params("hp_b1_2p3_periodic", 2.3),
    "chaotic": hp_dimensionless_params("hp_b1_3p0_chaotic", 3.0),
}


def domain_arrays(domain=DEFAULT_DOMAIN):
    domain_arr = np.asarray(domain, dtype=np.float64)
    lows = domain_arr[:, 0]
    highs = domain_arr[:, 1]
    spans = highs - lows
    if np.any(spans <= 0):
        raise ValueError("Domain upper bounds must exceed lower bounds")
    return lows, highs, spans


def unit_to_ecology(unit_points, domain=DEFAULT_DOMAIN):
    lows, _, spans = domain_arrays(domain)
    return lows + np.asarray(unit_points, dtype=np.float64) * spans


def ecology_to_unit(ecology_points, domain=DEFAULT_DOMAIN):
    lows, _, spans = domain_arrays(domain)
    return (np.asarray(ecology_points, dtype=np.float64) - lows) / spans


def holling_rhs_ecology(points, params: HollingParams):
    pts = np.asarray(points, dtype=np.float64)
    p = np.maximum(pts[..., 0], 0.0)
    h = np.maximum(pts[..., 1], 0.0)
    c = np.maximum(pts[..., 2], 0.0)

    uptake_h = params.alpha_h * params.beta_h * p / (params.gamma_h + p + 1e-12)
    uptake_c = params.alpha_c * params.beta_c * h / (params.gamma_c + h + 1e-12)

    dp = p * (1.0 - p / params.K) - uptake_h * h
    dh = h * (uptake_h - params.d_h) - uptake_c * c
    dc = c * (uptake_c - params.d_c)
    return np.stack([dp, dh, dc], axis=-1)


def holling_rhs_unit(unit_points, params: HollingParams, domain=DEFAULT_DOMAIN):
    eco = unit_to_ecology(unit_points, domain=domain)
    rhs_eco = holling_rhs_ecology(eco, params)
    _, _, spans = domain_arrays(domain)
    return rhs_eco / spans


def make_unit_grid(grid_size: int):
    axes = [np.linspace(0.0, 1.0, int(grid_size), dtype=np.float32) for _ in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1)


def vector_field_tensor(params: HollingParams, grid_size: int, domain=DEFAULT_DOMAIN):
    grid = make_unit_grid(grid_size)
    field = holling_rhs_unit(grid, params, domain=domain).astype(np.float32)
    return field


def tensor_to_torch_channels(field: np.ndarray):
    return np.asarray(field, dtype=np.float32).transpose(3, 0, 1, 2)


def torch_channels_to_tensor(channels) -> np.ndarray:
    arr = np.asarray(channels, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[0]
    return arr.transpose(1, 2, 3, 0)


def sample_alpha_family(seed: int, count: int) -> list[HollingParams]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(count)):
        out.append(
            HollingParams(
                name=f"alpha_sample_{i:05d}",
                K=float(rng.uniform(0.75, 1.25)),
                alpha_h=float(rng.uniform(0.20, 0.75)),
                alpha_c=float(rng.uniform(0.03, 0.18)),
                beta_h=2.009,
                beta_c=2.876,
                gamma_h=0.16129,
                gamma_c=0.5,
                d_h=0.4,
                d_c=0.08,
            )
        )
    return out


def jitter_params(
    params: HollingParams, rng: np.random.Generator, jitter: float = 0.10
) -> HollingParams:
    """Small multiplicative parameter perturbation for parent-only CAE training."""

    def mult(x: float) -> float:
        return float(x * rng.uniform(1.0 - jitter, 1.0 + jitter))

    return HollingParams(
        name=f"{params.name}_jitter",
        K=mult(params.K),
        alpha_h=mult(params.alpha_h),
        alpha_c=mult(params.alpha_c),
        beta_h=mult(params.beta_h),
        beta_c=mult(params.beta_c),
        gamma_h=mult(params.gamma_h),
        gamma_c=mult(params.gamma_c),
        d_h=mult(params.d_h),
        d_c=mult(params.d_c),
    )


def parent_augmented_samples(
    parents: list[HollingParams],
    seed: int,
    count: int,
    jitter: float = 0.10,
) -> list[HollingParams]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(count)):
        base = parents[i % len(parents)]
        if i < len(parents):
            out.append(base)
        else:
            out.append(jitter_params(base, rng, jitter=jitter))
    return out


def scan_alpha_grid() -> Iterable[HollingParams]:
    ks = [0.8, 0.9, 1.0, 1.1, 1.2]
    alpha_hs = [0.25, 0.35, 0.4, 0.5, 0.65]
    alpha_cs = [0.04, 0.06, 0.08, 0.10, 0.14]
    for K, alpha_h, alpha_c in product(ks, alpha_hs, alpha_cs):
        yield HollingParams(
            name=f"alpha_K{K:.2f}_ah{alpha_h:.2f}_ac{alpha_c:.2f}".replace(".", "p"),
            K=float(K),
            alpha_h=float(alpha_h),
            alpha_c=float(alpha_c),
        )
