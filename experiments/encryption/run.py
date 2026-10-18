from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, data, transform

from ndtl.systems import chen, lorenz
from ndtl.fields import calc_grids, ensemble_rescaling, vec2ode

from ndtl.paths import ROOT as PROJECT_ROOT

DEFAULT_VEC_PATH = PROJECT_ROOT / "outputs/encryption/fields/vecs.npy"
DEFAULT_FUSION_SUMMARY_ROOT = PROJECT_ROOT / "experiments/classic/results/content_style"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/encryption"
GRID_SHAPE = [48, 48, 48]
GRID_LIMITS = [[0, 1], [0, 1], [0, 1]]
IDEAL_NPCR = 99.609375
IDEAL_UACI = 33.4635


@dataclass
class SourceSpec:
    name: str
    kind: str
    ode: object
    candidate_index: int | None = None
    lyapunov_max_prior: float | None = None


@dataclass
class CryptoKey:
    permutation: np.ndarray
    mask: np.ndarray
    stream_values: np.ndarray
    mask_values: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Lorenz-Chen NDTL fused systems as image-encryption chaotic sources."
    )
    parser.add_argument("--vec-path", type=Path, default=DEFAULT_VEC_PATH)
    parser.add_argument(
        "--fusion-summary-root", type=Path, default=DEFAULT_FUSION_SUMMARY_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--image-names", nargs="+", default=["camera", "moon", "astronaut"]
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--candidate-indices", nargs="*", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--init", nargs=3, type=float, default=[0.3, 0.3, 0.5])
    parser.add_argument("--key-perturb", type=float, default=1e-14)
    parser.add_argument("--lyap-steps", type=int, default=5000)
    parser.add_argument("--lyap-transient", type=int, default=1000)
    parser.add_argument("--skip-parent-lyap", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small deterministic run for implementation checks.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def write_csv(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_standard_image(name: str, size: int) -> np.ndarray:
    if not hasattr(data, name):
        raise ValueError(f"Unknown skimage.data image: {name}")
    img = getattr(data, name)()
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = color.rgb2gray(arr)
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (size, size):
        arr = transform.resize(
            arr,
            (size, size),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
    return arr


def build_sources(
    vec_path: Path, summary_root: Path, candidate_indices: Iterable[int] | None
) -> list[SourceSpec]:
    _, lorenz_unit_nv = ensemble_rescaling(
        lorenz, target_grid_limits=GRID_LIMITS, target_velocity=5.0
    )
    _, chen_unit_nv = ensemble_rescaling(
        chen, target_grid_limits=GRID_LIMITS, target_velocity=5.0
    )
    sources = [
        SourceSpec(
            name="lorenz_unit_nv",
            kind="parent",
            ode=lorenz_unit_nv,
            candidate_index=None,
        ),
        SourceSpec(
            name="chen_unit_nv", kind="parent", ode=chen_unit_nv, candidate_index=None
        ),
    ]

    vecs = np.load(vec_path)
    if vecs.ndim != 6 or vecs.shape[1] != 1 or vecs.shape[-1] != 3:
        raise ValueError(
            f"Unexpected fused vector shape {vecs.shape}; expected (K,1,D,H,W,3)."
        )
    indices = (
        list(range(vecs.shape[0]))
        if candidate_indices is None
        else list(candidate_indices)
    )
    grids = calc_grids(grid_limits=GRID_LIMITS, grid_shape=GRID_SHAPE)
    for idx in indices:
        if idx < 0 or idx >= vecs.shape[0]:
            raise ValueError(
                f"Candidate index {idx} out of range for {vecs.shape[0]} candidates."
            )
        ode = vec2ode(
            vec=vecs[idx, 0],
            grids=grids,
            mode="linear",
            fit_intercept=True,
            name=f"fusion_lorenz_c_chen_s_vec_{idx}",
            grid_limits=GRID_LIMITS,
        )
        sources.append(
            SourceSpec(
                name=f"fusion_vec_{idx}",
                kind="fusion",
                ode=ode,
                candidate_index=idx,
                lyapunov_max_prior=load_fusion_lyapunov(summary_root, idx),
            )
        )
    return sources


def load_fusion_lyapunov(summary_root: Path, idx: int) -> float | None:
    path = summary_root / f"vec_{idx}" / "features_summary.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data_obj = json.load(f)
        spectrum = data_obj.get("lyapunov_spectrum_mean")
        if spectrum:
            return float(spectrum[0])
    except Exception:
        return None
    return None


def basis_degree2(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array(
        [1.0, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z], dtype=np.float64
    )


def rhs_degree2(ode, state: np.ndarray) -> np.ndarray:
    coef = np.asarray(ode.coef, dtype=np.float64)
    if coef.shape[1] < 10:
        raise ValueError(
            f"ODE {getattr(ode, 'name', 'ode')} has unsupported coef shape {coef.shape}."
        )
    return coef[:, :10] @ basis_degree2(state)


def jacobian_degree2(ode, state: np.ndarray) -> np.ndarray:
    coef = np.asarray(ode.coef, dtype=np.float64)[:, :10]
    x, y, z = state
    db_dx = np.array([0.0, 1.0, 0.0, 0.0, 2.0 * x, y, z, 0.0, 0.0, 0.0])
    db_dy = np.array([0.0, 0.0, 1.0, 0.0, 0.0, x, 0.0, 2.0 * y, z, 0.0])
    db_dz = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, x, 0.0, y, 2.0 * z])
    return np.stack([coef @ db_dx, coef @ db_dy, coef @ db_dz], axis=1)


def rk4_step(ode, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs_degree2(ode, state)
    k2 = rhs_degree2(ode, state + 0.5 * dt * k1)
    k3 = rhs_degree2(ode, state + 0.5 * dt * k2)
    k4 = rhs_degree2(ode, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_max_lyapunov_degree2(
    ode,
    init: np.ndarray,
    dt: float,
    transient: int = 1000,
    steps: int = 5000,
    max_abs: float = 1e6,
) -> float | None:
    state = np.asarray(init, dtype=np.float64).reshape(3)
    for _ in range(int(transient)):
        state = rk4_step(ode, state, dt)
        if (not np.all(np.isfinite(state))) or np.max(np.abs(state)) > max_abs:
            return None

    q = np.eye(3, dtype=np.float64)
    logs = np.zeros(3, dtype=np.float64)
    used = 0
    for _ in range(int(steps)):
        j = jacobian_degree2(ode, state)
        state = rk4_step(ode, state, dt)
        q = q + dt * (j @ q)
        q, r = np.linalg.qr(q)
        logs += np.log(np.abs(np.diag(r)) + 1e-300)
        used += 1
        if (not np.all(np.isfinite(state))) or np.max(np.abs(state)) > max_abs:
            return None
    if used == 0:
        return None
    spectrum = np.sort(logs / (dt * used))[::-1]
    return float(spectrum[0])


def integrate_stream_values(
    ode,
    init: np.ndarray,
    needed_values: int,
    burn_in: int,
    dt: float,
    max_abs: float = 1e6,
) -> tuple[np.ndarray, str]:
    steps = int(math.ceil(needed_values / 3.0)) + burn_in + 8
    state = np.asarray(init, dtype=np.float64).reshape(3)
    out = np.empty((max(0, steps - burn_in), 3), dtype=np.float64)
    j = 0
    for step in range(steps):
        state = rk4_step(ode, state, dt)
        if (not np.all(np.isfinite(state))) or np.max(np.abs(state)) > max_abs:
            return (
                np.array([], dtype=np.float64),
                f"failed: trajectory diverged at step {step}",
            )
        if step >= burn_in:
            out[j] = state
            j += 1
    flat = out.reshape(-1)
    return flat[:needed_values], "ok"


def values_to_unit_interval(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = np.abs(vals) * 1e8
    vals = vals - np.floor(vals)
    return vals.astype(np.float64)


def build_crypto_key(
    ode, image_shape: tuple[int, int], init: np.ndarray, burn_in: int, dt: float
) -> tuple[CryptoKey | None, dict]:
    n = int(np.prod(image_shape))
    values, status = integrate_stream_values(
        ode=ode,
        init=init,
        needed_values=2 * n,
        burn_in=burn_in,
        dt=dt,
    )
    if status != "ok":
        return None, {"status": status}
    unit = values_to_unit_interval(values)
    perm_values = unit[:n]
    mask_values = unit[n : 2 * n]
    permutation = np.argsort(perm_values, kind="mergesort")
    mask = np.floor(mask_values * 256.0).astype(np.uint8)
    return CryptoKey(
        permutation=permutation, mask=mask, stream_values=unit, mask_values=mask_values
    ), {"status": "ok"}


def encrypt_with_key(img: np.ndarray, key: CryptoKey) -> np.ndarray:
    flat = np.asarray(img, dtype=np.uint8).reshape(-1)
    permuted = flat[key.permutation]
    forward = np.empty_like(permuted)
    n = permuted.size
    state = 0xA5A5_1234
    for i in range(permuted.size):
        dyn = int(key.mask[(i + (state % n)) % n])
        state_byte = (state >> 16) & 0xFF
        forward[i] = np.uint8(
            (int(permuted[i]) + int(key.mask[i]) + dyn + state_byte) & 0xFF
        )
        state = update_diffusion_state(state, int(forward[i]), i, key.mask, n)

    cipher = np.empty_like(forward)
    state = 0x3C3C_5678
    for i in range(forward.size - 1, -1, -1):
        dyn = int(key.mask[(i + (state % n)) % n])
        state_byte = (state >> 16) & 0xFF
        cipher[i] = np.uint8(
            (int(forward[i]) + int(key.mask[i]) + dyn + state_byte) & 0xFF
        )
        state = update_diffusion_state(state, int(cipher[i]), i, key.mask, n)
    return cipher.reshape(img.shape)


def decrypt_with_key(cipher: np.ndarray, key: CryptoKey) -> np.ndarray:
    cflat = np.asarray(cipher, dtype=np.uint8).reshape(-1)
    n = cflat.size
    forward = np.empty_like(cflat)
    state = 0x3C3C_5678
    for i in range(cflat.size - 1, -1, -1):
        dyn = int(key.mask[(i + (state % n)) % n])
        state_byte = (state >> 16) & 0xFF
        cval = int(cflat[i])
        forward[i] = np.uint8((cval - int(key.mask[i]) - dyn - state_byte) & 0xFF)
        state = update_diffusion_state(state, cval, i, key.mask, n)

    permuted = np.empty_like(forward)
    state = 0xA5A5_1234
    for i in range(forward.size):
        dyn = int(key.mask[(i + (state % n)) % n])
        state_byte = (state >> 16) & 0xFF
        fval = int(forward[i])
        permuted[i] = np.uint8((fval - int(key.mask[i]) - dyn - state_byte) & 0xFF)
        state = update_diffusion_state(state, fval, i, key.mask, n)
    plain = np.empty_like(permuted)
    plain[key.permutation] = permuted
    return plain.reshape(cipher.shape)


def update_diffusion_state(
    state: int, cipher_value: int, i: int, mask: np.ndarray, n: int
) -> int:
    tap = int(mask[(17 * i + (state % n) + cipher_value) % n])
    mixed = state ^ (
        cipher_value
        + 0x9E37_79B9
        + ((state << 6) & 0xFFFF_FFFF)
        + (state >> 2)
        + i
        + tap
    )
    return (1664525 * mixed + 1013904223 + (tap << 8)) & 0xFFFF_FFFF


def image_entropy(img: np.ndarray) -> float:
    counts = np.bincount(
        np.asarray(img, dtype=np.uint8).reshape(-1), minlength=256
    ).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def hist_chi_square_uniform(img: np.ndarray) -> float:
    counts = np.bincount(
        np.asarray(img, dtype=np.uint8).reshape(-1), minlength=256
    ).astype(np.float64)
    expected = counts.sum() / 256.0
    return float(np.sum((counts - expected) ** 2 / (expected + 1e-12)))


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def adjacent_correlations(img: np.ndarray) -> dict[str, float]:
    x = np.asarray(img, dtype=np.uint8)
    h = corrcoef_safe(x[:, :-1], x[:, 1:])
    v = corrcoef_safe(x[:-1, :], x[1:, :])
    d = corrcoef_safe(x[:-1, :-1], x[1:, 1:])
    vals = [abs(c) for c in (h, v, d) if np.isfinite(c)]
    return {
        "adj_corr_h": h,
        "adj_corr_v": v,
        "adj_corr_d": d,
        "adj_corr_abs_mean": float(np.mean(vals)) if vals else float("nan"),
    }


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa = np.asarray(a, dtype=np.uint8)
    bb = np.asarray(b, dtype=np.uint8)
    if aa.shape != bb.shape:
        raise ValueError(f"Shape mismatch for NPCR/UACI: {aa.shape} vs {bb.shape}")
    diff = aa.astype(np.int16) - bb.astype(np.int16)
    npcr = 100.0 * float(np.mean(aa != bb))
    uaci = 100.0 * float(np.mean(np.abs(diff)) / 255.0)
    return npcr, uaci


def approximate_entropy_bits(bits: np.ndarray, m: int = 2) -> float:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = b.size
    if n <= m + 1:
        return float("nan")

    def phi(mm: int) -> float:
        padded = np.concatenate([b, b[: mm - 1]])
        counts = np.zeros(2**mm, dtype=np.float64)
        for i in range(n):
            idx = 0
            for bit in padded[i : i + mm]:
                idx = (idx << 1) | int(bit)
            counts[idx] += 1.0
        probs = counts / n
        probs = probs[probs > 0]
        return float(np.sum(probs * np.log(probs)))

    return float(phi(m) - phi(m + 1))


def sequence_metrics(key: CryptoKey) -> dict[str, float]:
    mask = key.mask.astype(np.uint8)
    stream = key.stream_values.astype(np.float64)
    counts = np.bincount(mask, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    byte_entropy = float(-np.sum(p * np.log2(p)))
    autocorr = corrcoef_safe(stream[:-1], stream[1:])
    bits = np.unpackbits(mask[: min(mask.size, 4096)])
    return {
        "keystream_byte_entropy": byte_entropy,
        "keystream_lag1_abs_corr": (
            abs(autocorr) if np.isfinite(autocorr) else float("nan")
        ),
        "keystream_approx_entropy_bits_m2": approximate_entropy_bits(bits, m=2),
    }


def metric_scores(row: dict) -> dict[str, float]:
    entropy_score = max(0.0, 1.0 - abs(float(row["cipher_entropy"]) - 8.0) / 8.0)
    corr_score = max(0.0, 1.0 - min(abs(float(row["adj_corr_abs_mean"])), 1.0))
    npcr_score = max(0.0, 1.0 - abs(float(row["npcr"]) - IDEAL_NPCR) / IDEAL_NPCR)
    uaci_score = max(0.0, 1.0 - abs(float(row["uaci"]) - IDEAL_UACI) / IDEAL_UACI)
    key_npcr_score = max(
        0.0, 1.0 - abs(float(row["key_npcr"]) - IDEAL_NPCR) / IDEAL_NPCR
    )
    key_uaci_score = max(
        0.0, 1.0 - abs(float(row["key_uaci"]) - IDEAL_UACI) / IDEAL_UACI
    )
    score = float(
        np.mean(
            [
                entropy_score,
                corr_score,
                npcr_score,
                uaci_score,
                key_npcr_score,
                key_uaci_score,
            ]
        )
    )
    return {
        "entropy_score": entropy_score,
        "corr_score": corr_score,
        "npcr_score": npcr_score,
        "uaci_score": uaci_score,
        "key_npcr_score": key_npcr_score,
        "key_uaci_score": key_uaci_score,
        "crypto_score": score,
    }


def altered_plain(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    out.flat[0] = np.uint8(out.flat[0] ^ 1)
    return out


def save_image(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def plot_diagnostic(
    path: Path, plain: np.ndarray, cipher: np.ndarray, title: str
) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), dpi=180)
    axes[0].imshow(plain, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Plain")
    axes[1].imshow(cipher, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Cipher")
    axes[2].hist(cipher.reshape(-1), bins=256, range=(0, 255), color="#555555")
    axes[2].set_title("Cipher histogram")
    sample_x = cipher[:, :-1].reshape(-1)
    sample_y = cipher[:, 1:].reshape(-1)
    if sample_x.size > 8000:
        idx = np.linspace(0, sample_x.size - 1, 8000).astype(int)
        sample_x = sample_x[idx]
        sample_y = sample_y[idx]
    axes[3].scatter(sample_x, sample_y, s=1, alpha=0.25)
    axes[3].set_title("Horizontal pairs")
    axes[3].set_xlim(0, 255)
    axes[3].set_ylim(0, 255)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_summary(path: Path, summary_rows: list[dict]) -> None:
    ensure_dir(path.parent)
    labels = [r["source_name"] for r in summary_rows if r.get("status") == "ok"]
    scores = [
        float(r["crypto_score_mean"]) for r in summary_rows if r.get("status") == "ok"
    ]
    colors = [
        "#4c78a8" if not label.startswith("fusion") else "#f58518" for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 3.8), dpi=180)
    ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean crypto score")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize_rows(rows: list[dict], source_rows: list[dict]) -> list[dict]:
    out = []
    names = sorted({r["source_name"] for r in source_rows})
    for name in names:
        source_info = next(r for r in source_rows if r["source_name"] == name)
        matching = [
            r for r in rows if r["source_name"] == name and r.get("status") == "ok"
        ]
        if not matching:
            out.append({**source_info, "status": source_info.get("status", "failed")})
            continue
        numeric_keys = [
            "cipher_entropy",
            "hist_chi_square",
            "adj_corr_abs_mean",
            "npcr",
            "uaci",
            "key_npcr",
            "key_uaci",
            "wrong_decrypt_npcr",
            "wrong_decrypt_uaci",
            "crypto_score",
        ]
        summary = {**source_info, "status": "ok", "num_images": len(matching)}
        for key in numeric_keys:
            vals = [float(r[key]) for r in matching if np.isfinite(float(r[key]))]
            summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(summary)
    return out


def compare_winner(summary_rows: list[dict]) -> dict:
    ok = [r for r in summary_rows if r.get("status") == "ok"]
    parents = {r["source_name"]: r for r in ok if r.get("source_kind") == "parent"}
    fusions = [r for r in ok if r.get("source_kind") == "fusion"]
    if not fusions:
        return {"status": "no_successful_fusion"}
    winner = max(fusions, key=lambda r: float(r["crypto_score_mean"]))
    parent_names = ["lorenz_unit_nv", "chen_unit_nv"]
    beats = {}
    core_majority = {}
    for pname in parent_names:
        p = parents.get(pname)
        if p is None:
            beats[pname] = False
            core_majority[pname] = 0
            continue
        beats[pname] = float(winner["crypto_score_mean"]) > float(
            p["crypto_score_mean"]
        )
        count = 0
        winner_sensitivity_gap = 0.5 * (
            abs(float(winner["npcr_mean"]) - IDEAL_NPCR) / IDEAL_NPCR
            + abs(float(winner["uaci_mean"]) - IDEAL_UACI) / IDEAL_UACI
        )
        parent_sensitivity_gap = 0.5 * (
            abs(float(p["npcr_mean"]) - IDEAL_NPCR) / IDEAL_NPCR
            + abs(float(p["uaci_mean"]) - IDEAL_UACI) / IDEAL_UACI
        )
        count += winner_sensitivity_gap < parent_sensitivity_gap
        count += abs(float(winner["cipher_entropy_mean"]) - 8.0) < abs(
            float(p["cipher_entropy_mean"]) - 8.0
        )
        count += float(winner["adj_corr_abs_mean_mean"]) < float(
            p["adj_corr_abs_mean_mean"]
        )
        core_majority[pname] = int(count)
    return {
        "status": "ok",
        "winner": winner,
        "beats_parent_score": beats,
        "core_metric_wins_out_of_3": core_majority,
        "passes_claim_gate": bool(
            all(beats.values()) and all(v >= 2 for v in core_majority.values())
        ),
        "claim_gate_note": "Fusion must beat both parents by mean score and win at least 2 of 3 core dimensions: NPCR/UACI sensitivity, entropy, adjacent correlation.",
    }


def run_benchmark(args) -> dict:
    if args.smoke:
        args.image_size = min(args.image_size, 64)
        args.image_names = args.image_names[:1]
        if args.candidate_indices is None:
            args.candidate_indices = [6]

    ensure_dir(args.out_dir)
    images = {
        name: load_standard_image(name, args.image_size) for name in args.image_names
    }
    sources = build_sources(
        args.vec_path, args.fusion_summary_root, args.candidate_indices
    )
    max_shape = next(iter(images.values())).shape
    init = np.asarray(args.init, dtype=np.float64)
    if args.smoke:
        args.lyap_steps = min(args.lyap_steps, 800)
        args.lyap_transient = min(args.lyap_transient, 200)
    if not args.skip_parent_lyap:
        for source in sources:
            if source.lyapunov_max_prior is None and source.kind == "parent":
                source.lyapunov_max_prior = estimate_max_lyapunov_degree2(
                    source.ode,
                    init=init,
                    dt=args.dt,
                    transient=args.lyap_transient,
                    steps=args.lyap_steps,
                )
    perturbed_init = init.copy()
    perturbed_init[0] += float(args.key_perturb)

    config = {
        "vec_path": args.vec_path,
        "fusion_summary_root": args.fusion_summary_root,
        "out_dir": args.out_dir,
        "image_names": args.image_names,
        "image_size": args.image_size,
        "dt": args.dt,
        "burn_in": args.burn_in,
        "init": init,
        "key_perturb": args.key_perturb,
        "ideal_npcr": IDEAL_NPCR,
        "ideal_uaci": IDEAL_UACI,
        "sources": [
            {
                "source_name": s.name,
                "source_kind": s.kind,
                "candidate_index": s.candidate_index,
                "lyapunov_max_prior": s.lyapunov_max_prior,
            }
            for s in sources
        ],
    }
    save_json(args.out_dir / "config.json", config)

    per_image_rows: list[dict] = []
    source_sequence_rows: list[dict] = []
    key_cache: dict[str, CryptoKey] = {}
    perturb_key_cache: dict[str, CryptoKey] = {}

    for source in sources:
        print(f"[source] {source.name}")
        key, key_meta = build_crypto_key(
            source.ode, max_shape, init, args.burn_in, args.dt
        )
        pkey, pkey_meta = build_crypto_key(
            source.ode, max_shape, perturbed_init, args.burn_in, args.dt
        )
        source_row = {
            "source_name": source.name,
            "source_kind": source.kind,
            "candidate_index": source.candidate_index,
            "lyapunov_max_prior": source.lyapunov_max_prior,
            "status": key_meta["status"],
            "perturbed_status": pkey_meta["status"],
        }
        if key is None or pkey is None:
            source_sequence_rows.append(source_row)
            continue

        key_cache[source.name] = key
        perturb_key_cache[source.name] = pkey
        source_sequence_rows.append({**source_row, **sequence_metrics(key)})

        for image_name, img in images.items():
            cipher = encrypt_with_key(img, key)
            plain_hat = decrypt_with_key(cipher, key)
            deterministic_cipher = encrypt_with_key(img, key)
            alt_cipher = encrypt_with_key(altered_plain(img), key)
            perturb_cipher = encrypt_with_key(img, pkey)
            wrong_decrypt = decrypt_with_key(cipher, pkey)
            npcr, uaci = npcr_uaci(cipher, alt_cipher)
            key_npcr, key_uaci = npcr_uaci(cipher, perturb_cipher)
            wrong_npcr, wrong_uaci = npcr_uaci(img, wrong_decrypt)
            corr = adjacent_correlations(cipher)
            row = {
                "source_name": source.name,
                "source_kind": source.kind,
                "candidate_index": source.candidate_index,
                "image_name": image_name,
                "status": "ok",
                "roundtrip_ok": bool(np.array_equal(img, plain_hat)),
                "deterministic_ok": bool(np.array_equal(cipher, deterministic_cipher)),
                "plain_entropy": image_entropy(img),
                "cipher_entropy": image_entropy(cipher),
                "hist_chi_square": hist_chi_square_uniform(cipher),
                **corr,
                "npcr": npcr,
                "uaci": uaci,
                "key_npcr": key_npcr,
                "key_uaci": key_uaci,
                "wrong_decrypt_npcr": wrong_npcr,
                "wrong_decrypt_uaci": wrong_uaci,
            }
            row.update(metric_scores(row))
            per_image_rows.append(row)

            image_dir = args.out_dir / "images" / source.name
            save_image(image_dir / f"{image_name}_plain.png", img)
            save_image(image_dir / f"{image_name}_cipher.png", cipher)
            save_image(image_dir / f"{image_name}_wrong_key_decrypt.png", wrong_decrypt)
            if not args.no_figures:
                plot_diagnostic(
                    args.out_dir
                    / "figures"
                    / source.name
                    / f"{image_name}_diagnostic.png",
                    img,
                    cipher,
                    f"{source.name} / {image_name}",
                )

    summary_rows = summarize_rows(per_image_rows, source_sequence_rows)
    winner_report = compare_winner(summary_rows)

    write_csv(args.out_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(args.out_dir / "source_sequence_metrics.csv", source_sequence_rows)
    write_csv(args.out_dir / "source_summary.csv", summary_rows)
    save_json(args.out_dir / "winner_report.json", winner_report)
    if not args.no_figures:
        plot_summary(args.out_dir / "figures" / "summary_scores.png", summary_rows)

    report = {
        "config": config,
        "summary_rows": summary_rows,
        "winner_report": winner_report,
        "outputs": {
            "per_image_metrics": str(args.out_dir / "per_image_metrics.csv"),
            "source_sequence_metrics": str(
                args.out_dir / "source_sequence_metrics.csv"
            ),
            "source_summary": str(args.out_dir / "source_summary.csv"),
            "winner_report": str(args.out_dir / "winner_report.json"),
        },
    }
    save_json(args.out_dir / "report.json", report)
    return report


def main():
    args = parse_args()
    report = run_benchmark(args)
    winner = report["winner_report"].get("winner", {})
    if winner:
        print(
            "winner=",
            winner.get("source_name"),
            "score=",
            winner.get("crypto_score_mean"),
            "passes_claim_gate=",
            report["winner_report"].get("passes_claim_gate"),
        )
    else:
        print("No successful fusion winner.")


if __name__ == "__main__":
    main()
