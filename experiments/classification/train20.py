from __future__ import annotations

from .arguments import add_shared_arguments

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from ndtl.paths import ROOT as _ROOT

LOCAL_CACHE_ROOT = _ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT / "xdg"))

import numpy as np
import torch
import torch.nn.functional as F

from ndtl.paths import ROOT as PROJECT_ROOT, inside_project as resolve_inside_project

from ndtl.models import CAE  # noqa: E402
from experiments.classification.systems import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_INIT_POINT,
    TARGET_LIMITS,
    TARGET_VELOCITY,
    LyapConfig,
    build_pair_component_rows,
    build_system_specs,
    clone_ode,
    compute_lyapunov_light,
    format_csv_value,
    matrix_from_pair_rows,
    parse_alpha_list,
    spectra_from_records,
    write_matrix_csv,
    write_nearest_neighbors,
    write_pair_components,
    write_records_csv,
)
from ndtl.metrics import save_json  # noqa: E402
from ndtl.fields import (
    ODE,
    calc_grids,
    ensemble_rescaling,
    ode2vec,
    tensor2vec,
    vec2ode,
)  # noqa: E402

DEFAULT_RESULTS_DIR = Path("outputs/classification/systems20")
SMOKE_RESULTS_DIR = Path("outputs/classification/systems20_smoke")
DEFAULT_HOMOTOPY_RESULTS_DIR = Path("experiments/classification/configs/systems20")
PAIR_COEF_DIRNAME = "pair_coefficients"


@dataclass(frozen=True)
class CaeConfig:
    grid_shape: tuple[int, int, int] = (48, 48, 48)
    depth: int = 64
    strides: tuple[int, int, int] = (3, 2, 2)
    kernel_size: tuple[int, int, int] = (5, 3, 3)
    batch_size: int = 12
    max_batches: int = 5000
    lr: float = 1e-3
    turb_scale: float = 0.1
    reduce_factor: float = 0.8
    reduce_patience: int = 100
    early_lr: float = 1e-4
    log_every: int = 500


@dataclass(frozen=True)
class TransferConfig:
    max_iters: int = 10000
    lr: float = 0.1
    reduce_factor: float = 0.5
    reduce_patience: int = 500
    early_lr: float = 1e-6
    log_every: int = 1000


def setup_logger(log_path: Path, name: str = "content_only_paircae") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def parse_int_tuple(text: str, expected: int = 3) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated integers, got {text!r}")
    return values


def pair_id(name_a: str, name_b: str) -> str:
    return f"{name_a}__{name_b}"


def pair_coef_path(out_root: Path, pair_index: int, name_a: str, name_b: str) -> Path:
    return (
        out_root / PAIR_COEF_DIRNAME / f"{pair_index:03d}_{pair_id(name_a, name_b)}.npz"
    )


def shard_name(args: argparse.Namespace) -> str:
    if args.pair_shard_count is None:
        return "single"
    return f"shard_{int(args.pair_shard_index):02d}_of_{int(args.pair_shard_count):02d}"


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def safe_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_csv_value(row.get(k, "")) for k in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


EVAL_FIELDNAMES_SHARD = [
    "pair_index",
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

CAE_FIELDNAMES = [
    "pair_index",
    "pair_id",
    "system_a",
    "system_b",
    "status",
    "train_batches",
    "best_loss",
    "final_loss",
    "final_lr",
    "recon_mse_a",
    "recon_mse_b",
    "model_path",
    "elapsed_sec",
    "error",
]

FUSION_FIELDNAMES = [
    "pair_index",
    "pair_id",
    "system_a",
    "system_b",
    "alpha_index",
    "alpha",
    "status",
    "iterations",
    "best_loss",
    "final_loss",
    "final_lr",
    "content_loss_a",
    "content_loss_b",
    "regression_mse_mean",
    "regression_mae_mean",
    "regression_r2_mean",
    "regression_r2_min",
    "coef_path",
    "elapsed_sec",
    "error",
]


def parse_eval_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "pair_id": row["pair_id"],
        "system_a": row["system_a"],
        "system_b": row["system_b"],
        "alpha_index": int(row["alpha_index"]),
        "alpha": float(row["alpha"]),
        "status": row["status"],
        "lambda1": safe_float(row.get("lambda1")),
        "lambda2": safe_float(row.get("lambda2")),
        "lambda3": safe_float(row.get("lambda3")),
        "used_steps": safe_int(row.get("used_steps")),
        "dt": safe_float(row.get("dt")),
        "error": row.get("error", ""),
    }
    if "pair_index" in row and row["pair_index"] != "":
        parsed["pair_index"] = int(row["pair_index"])
    return parsed


def load_shard_eval_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv_rows(path):
        parsed = parse_eval_row(row)
        out[(str(parsed["pair_id"]), int(parsed["alpha_index"]))] = parsed
    return out


def load_shard_cae_pair_ids(path: Path) -> set[str]:
    rows = read_csv_rows(path)
    return {row["pair_id"] for row in rows if row.get("status") == "OK"}


def load_homotopy_bbox(homotopy_results_dir: Path) -> dict[str, dict[str, Any]]:
    bbox_path = homotopy_results_dir / "estimated_grid_limits.json"
    if not bbox_path.exists():
        raise FileNotFoundError(f"missing linear-homotopy bbox file: {bbox_path}")
    with bbox_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_systems_from_bbox(
    specs: list[Any],
    bbox_records: dict[str, dict[str, Any]],
    lyap_h: float,
) -> tuple[dict[str, ODE], dict[str, dict[str, Any]]]:
    normalized: dict[str, ODE] = {}
    out_records: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.key not in bbox_records:
            raise KeyError(f"missing bbox record for {spec.key}")
        limits = bbox_records[spec.key]["estimated_grid_limits"]
        ode_for_scaling = clone_ode(spec.ode, name=spec.key, grid_limits=limits)
        _, ode_norm = ensemble_rescaling(
            ode_for_scaling,
            target_grid_limits=TARGET_LIMITS,
            target_velocity=TARGET_VELOCITY,
        )
        ode_norm.name = spec.key
        ode_norm.h = float(lyap_h)
        normalized[spec.key] = ode_norm

        record = dict(bbox_records[spec.key])
        record["normalized_grid_limits"] = ode_norm.grid_limits
        record["target_velocity"] = TARGET_VELOCITY
        record["normalized_system_velocity"] = ode_norm.system_velocity
        record["lyap_h"] = ode_norm.h
        out_records[spec.key] = record
    return normalized, out_records


def make_tensor_from_ode(
    ode: ODE, grids: list[np.ndarray], device: torch.device
) -> torch.Tensor:
    vec = ode2vec(ode, grids)
    tensor = torch.tensor(vec, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor.to(device=device, non_blocking=True)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_cae(config: CaeConfig, device: torch.device) -> CAE:
    model = CAE(
        cae_depth=int(config.depth),
        cae_input_dimension=3,
        cae_strides=list(config.strides),
        cae_kernel_size=list(config.kernel_size),
    )
    return model.to(device)


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def random_pair_batch(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    batch_size: int,
    turb_scale: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    base = torch.cat([tensor_a, tensor_b], dim=0)
    indices = torch.randint(
        0, 2, (batch_size,), device=base.device, generator=generator
    )
    batch = base.index_select(dim=0, index=indices)
    if turb_scale > 0.0:
        noise = torch.randn(
            batch.shape,
            dtype=batch.dtype,
            device=batch.device,
            generator=generator,
        )
        batch = batch + float(turb_scale) * noise
    return batch


def train_or_load_pair_cae(
    pair_index: int,
    name_a: str,
    name_b: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    config: CaeConfig,
    model_path: Path,
    device: torch.device,
    seed: int,
    resume: bool,
    logger: logging.Logger,
) -> tuple[CAE, dict[str, Any]]:
    pid = pair_id(name_a, name_b)
    start = time.time()
    model = build_cae(config, device)
    status = "OK"
    error = ""
    train_batches = 0
    best_loss = float("nan")
    final_loss = float("nan")
    final_lr = float("nan")

    try:
        if resume and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            train_batches = int(checkpoint.get("train_batches", 0))
            best_loss = float(checkpoint.get("best_loss", float("nan")))
            final_loss = float(checkpoint.get("final_loss", float("nan")))
            final_lr = float(checkpoint.get("final_lr", float("nan")))
            logger.info("Loaded CAE for pair %03d %s", pair_index, pid)
        else:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(config.reduce_factor),
                patience=int(config.reduce_patience),
            )
            best_state: dict[str, torch.Tensor] | None = None
            best_loss = float("inf")
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + pair_index)

            for step in range(1, int(config.max_batches) + 1):
                batch = random_pair_batch(
                    tensor_a,
                    tensor_b,
                    int(config.batch_size),
                    float(config.turb_scale),
                    generator,
                )
                optimizer.zero_grad(set_to_none=True)
                _, _, recon = model(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                scheduler.step(loss_value)

                train_batches = step
                final_loss = loss_value
                final_lr = float(optimizer.param_groups[0]["lr"])
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_state = state_dict_to_cpu(model)

                if (
                    step % int(config.log_every) == 0
                    or step == 1
                    or step == int(config.max_batches)
                ):
                    logger.info(
                        "CAE pair %03d %s step %d/%d loss %.6g lr %.3g",
                        pair_index,
                        pid,
                        step,
                        int(config.max_batches),
                        loss_value,
                        final_lr,
                    )
                if final_lr <= float(config.early_lr):
                    logger.info(
                        "CAE pair %03d %s early stop at lr %.3g",
                        pair_index,
                        pid,
                        final_lr,
                    )
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "pair_index": pair_index,
                    "pair_id": pid,
                    "system_a": name_a,
                    "system_b": name_b,
                    "config": asdict(config),
                    "train_batches": train_batches,
                    "best_loss": best_loss,
                    "final_loss": final_loss,
                    "final_lr": final_lr,
                },
                model_path,
            )

        model.eval()
        with torch.no_grad():
            _, _, recon_a = model(tensor_a)
            _, _, recon_b = model(tensor_b)
            recon_mse_a = float(F.mse_loss(recon_a, tensor_a).detach().cpu().item())
            recon_mse_b = float(F.mse_loss(recon_b, tensor_b).detach().cpu().item())
    except Exception as exc:  # noqa: BLE001
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        recon_mse_a = float("nan")
        recon_mse_b = float("nan")

    row = {
        "pair_index": pair_index,
        "pair_id": pid,
        "system_a": name_a,
        "system_b": name_b,
        "status": status,
        "train_batches": train_batches,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "final_lr": final_lr,
        "recon_mse_a": recon_mse_a,
        "recon_mse_b": recon_mse_b,
        "model_path": relative_path(model_path),
        "elapsed_sec": time.time() - start,
        "error": error,
    }
    if status != "OK":
        raise RuntimeError(error)
    return model, row


@torch.no_grad()
def endpoint_features(model: CAE, tensor: torch.Tensor) -> list[torch.Tensor]:
    feats, _, _ = model(tensor)
    return [feat.detach().clone() for feat in feats]


def content_losses(
    feats: list[torch.Tensor],
    target_a: list[torch.Tensor],
    target_b: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    losses_a = [
        F.mse_loss(feat, target, reduction="mean")
        for feat, target in zip(feats, target_a)
    ]
    losses_b = [
        F.mse_loss(feat, target, reduction="mean")
        for feat, target in zip(feats, target_b)
    ]
    return torch.stack(losses_a).mean(), torch.stack(losses_b).mean()


def aggregate_regression_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    per_dim = diagnostics.get("per_dim", [])
    if not per_dim:
        return {
            "regression_mse_mean": float("nan"),
            "regression_mae_mean": float("nan"),
            "regression_r2_mean": float("nan"),
            "regression_r2_min": float("nan"),
        }
    mses = np.asarray([d.get("mse", np.nan) for d in per_dim], dtype=float)
    maes = np.asarray([d.get("mae", np.nan) for d in per_dim], dtype=float)
    r2s = np.asarray([d.get("r2", np.nan) for d in per_dim], dtype=float)
    return {
        "regression_mse_mean": float(np.nanmean(mses)),
        "regression_mae_mean": float(np.nanmean(maes)),
        "regression_r2_mean": float(np.nanmean(r2s)),
        "regression_r2_min": float(np.nanmin(r2s)),
    }


def fit_tensor_to_quadratic_ode(
    tensor: torch.Tensor,
    grids: list[np.ndarray],
    alpha: float,
    name_a: str,
    name_b: str,
    lyap_h: float,
) -> tuple[ODE, dict[str, float]]:
    vec = tensor2vec(tensor.detach().cpu())
    ode, diagnostics = vec2ode(
        vec,
        grids,
        max_order=2,
        extra_term_dicts={},
        mode="linear",
        name=f"content__{name_a}__{name_b}__alpha{alpha:.3f}",
        return_diagnostics=True,
        grid_limits=TARGET_LIMITS,
    )
    ode.h = float(lyap_h)
    return ode, aggregate_regression_diagnostics(diagnostics)


def run_content_transfer(
    pair_index: int,
    name_a: str,
    name_b: str,
    alpha_index: int,
    alpha: float,
    model: CAE,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    target_a: list[torch.Tensor],
    target_b: list[torch.Tensor],
    grids: list[np.ndarray],
    config: TransferConfig,
    lyap_h: float,
    logger: logging.Logger,
) -> tuple[ODE | None, dict[str, Any]]:
    pid = pair_id(name_a, name_b)
    start = time.time()
    status = "OK"
    error = ""
    iterations = 0
    best_loss = float("inf")
    final_loss = float("nan")
    final_lr = float("nan")
    final_content_a = float("nan")
    final_content_b = float("nan")
    reg = {
        "regression_mse_mean": float("nan"),
        "regression_mae_mean": float("nan"),
        "regression_r2_mean": float("nan"),
        "regression_r2_min": float("nan"),
    }
    ode_hat: ODE | None = None

    try:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        warm = (1.0 - float(alpha)) * tensor_a + float(alpha) * tensor_b
        fused = warm.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([fused], lr=float(config.lr))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.reduce_factor),
            patience=int(config.reduce_patience),
        )
        best_tensor = fused.detach().clone()

        for step in range(1, int(config.max_iters) + 1):
            optimizer.zero_grad(set_to_none=True)
            feats, _, _ = model(fused)
            loss_a, loss_b = content_losses(feats, target_a, target_b)
            loss = (1.0 - float(alpha)) * loss_a + float(alpha) * loss_b
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            scheduler.step(loss_value)
            iterations = step
            final_loss = loss_value
            final_lr = float(optimizer.param_groups[0]["lr"])
            final_content_a = float(loss_a.detach().cpu().item())
            final_content_b = float(loss_b.detach().cpu().item())

            if loss_value < best_loss:
                best_loss = loss_value
                best_tensor = fused.detach().clone()

            if (
                step % int(config.log_every) == 0
                or step == 1
                or step == int(config.max_iters)
            ):
                logger.info(
                    "Fusion pair %03d %s alpha %.3f step %d/%d loss %.6g lr %.3g",
                    pair_index,
                    pid,
                    float(alpha),
                    step,
                    int(config.max_iters),
                    loss_value,
                    final_lr,
                )
            if final_lr <= float(config.early_lr):
                logger.info(
                    "Fusion pair %03d %s alpha %.3f early stop at lr %.3g",
                    pair_index,
                    pid,
                    float(alpha),
                    final_lr,
                )
                break

        ode_hat, reg = fit_tensor_to_quadratic_ode(
            best_tensor,
            grids,
            float(alpha),
            name_a,
            name_b,
            lyap_h,
        )
    except Exception as exc:  # noqa: BLE001
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"

    row = {
        "pair_index": pair_index,
        "pair_id": pid,
        "system_a": name_a,
        "system_b": name_b,
        "alpha_index": alpha_index,
        "alpha": alpha,
        "status": status,
        "iterations": iterations,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "final_lr": final_lr,
        "content_loss_a": final_content_a,
        "content_loss_b": final_content_b,
        **reg,
        "coef_path": "",
        "elapsed_sec": time.time() - start,
        "error": error,
    }
    return ode_hat, row


def lyap_eval_row(
    pair_index: int,
    name_a: str,
    name_b: str,
    alpha_index: int,
    alpha: float,
    ode: ODE | None,
    lyap_cfg: LyapConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_index": pair_index,
        "pair_id": pair_id(name_a, name_b),
        "system_a": name_a,
        "system_b": name_b,
        "alpha_index": alpha_index,
        "alpha": float(alpha),
        "status": "OK",
        "lambda1": np.nan,
        "lambda2": np.nan,
        "lambda3": np.nan,
        "used_steps": 0,
        "dt": float(lyap_cfg.h),
        "error": "",
    }
    try:
        if ode is None:
            raise RuntimeError("missing ODE for Lyapunov evaluation")
        res = compute_lyapunov_light(ode, DEFAULT_INIT_POINT, lyap_cfg)
        spectrum = np.asarray(res["spectrum"], dtype=float).reshape(-1)
        if spectrum.size != 3 or not np.all(np.isfinite(spectrum)):
            raise ValueError(f"invalid spectrum: {spectrum}")
        row["lambda1"], row["lambda2"], row["lambda3"] = [float(v) for v in spectrum]
        row["used_steps"] = int(res["used_steps"])
        row["dt"] = float(res["dt"])
    except Exception as exc:  # noqa: BLE001
        row["status"] = "FAIL"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def save_pair_coefficients(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_one_pair(
    pair_index: int,
    name_a: str,
    name_b: str,
    normalized: dict[str, ODE],
    grids: list[np.ndarray],
    alphas: np.ndarray,
    cae_config: CaeConfig,
    transfer_config: TransferConfig,
    lyap_cfg: LyapConfig,
    out_root: Path,
    shard_eval: dict[tuple[str, int], dict[str, Any]],
    resume: bool,
    device: torch.device,
    seed: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pid = pair_id(name_a, name_b)
    model_path = out_root / "pair_models" / f"{pair_index:03d}_{pid}" / "cae.pt"
    coef_path = pair_coef_path(out_root, pair_index, name_a, name_b)
    existing_coef = load_npz_dict(coef_path) if resume else {}
    expected_keys = {(pid, int(i)) for i in range(len(alphas))}
    if resume and expected_keys.issubset(set(shard_eval)) and coef_path.exists():
        logger.info("Skip completed pair %03d %s", pair_index, pid)
        return {}, [], []

    tensor_a = make_tensor_from_ode(normalized[name_a], grids, device)
    tensor_b = make_tensor_from_ode(normalized[name_b], grids, device)
    model, cae_row = train_or_load_pair_cae(
        pair_index,
        name_a,
        name_b,
        tensor_a,
        tensor_b,
        cae_config,
        model_path,
        device,
        seed,
        resume,
        logger,
    )

    target_a = endpoint_features(model, tensor_a)
    target_b = endpoint_features(model, tensor_b)
    fusion_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    coef_arrays = dict(existing_coef)
    coef_arrays["alphas"] = np.asarray(alphas, dtype=float)

    for alpha_index, alpha_value in enumerate(alphas):
        alpha = float(alpha_value)
        key = (pid, int(alpha_index))
        if resume and key in shard_eval:
            continue

        if alpha_index == 0 or np.isclose(alpha, 0.0):
            ode_eval = clone_ode(
                normalized[name_a],
                name=f"endpoint__{name_a}",
                grid_limits=TARGET_LIMITS,
            )
            ode_eval.h = float(lyap_cfg.h)
            coef_arrays[f"alpha_{alpha_index:03d}_coef"] = np.asarray(
                ode_eval.coef, dtype=float
            )
            eval_rows.append(
                lyap_eval_row(
                    pair_index, name_a, name_b, alpha_index, alpha, ode_eval, lyap_cfg
                )
            )
            continue
        if alpha_index == len(alphas) - 1 or np.isclose(alpha, 1.0):
            ode_eval = clone_ode(
                normalized[name_b],
                name=f"endpoint__{name_b}",
                grid_limits=TARGET_LIMITS,
            )
            ode_eval.h = float(lyap_cfg.h)
            coef_arrays[f"alpha_{alpha_index:03d}_coef"] = np.asarray(
                ode_eval.coef, dtype=float
            )
            eval_rows.append(
                lyap_eval_row(
                    pair_index, name_a, name_b, alpha_index, alpha, ode_eval, lyap_cfg
                )
            )
            continue

        ode_hat, fusion_row = run_content_transfer(
            pair_index,
            name_a,
            name_b,
            alpha_index,
            alpha,
            model,
            tensor_a,
            tensor_b,
            target_a,
            target_b,
            grids,
            transfer_config,
            float(lyap_cfg.h),
            logger,
        )
        fusion_row["coef_path"] = relative_path(coef_path)
        fusion_rows.append(fusion_row)
        if ode_hat is not None:
            coef_arrays[f"alpha_{alpha_index:03d}_coef"] = np.asarray(
                ode_hat.coef, dtype=float
            )
        eval_rows.append(
            lyap_eval_row(
                pair_index, name_a, name_b, alpha_index, alpha, ode_hat, lyap_cfg
            )
        )
        save_pair_coefficients(coef_path, coef_arrays)

    save_pair_coefficients(coef_path, coef_arrays)
    del tensor_a, tensor_b, target_a, target_b, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cae_row, fusion_rows, eval_rows


def all_pairs(names: list[str]) -> list[tuple[int, str, str]]:
    return [(i, a, b) for i, (a, b) in enumerate(combinations(names, 2))]


def select_pairs_for_args(
    pairs: list[tuple[int, str, str]], args: argparse.Namespace
) -> list[tuple[int, str, str]]:
    selected = pairs
    if args.pair_start is not None:
        selected = [p for p in selected if p[0] >= int(args.pair_start)]
    if args.pair_end is not None:
        selected = [p for p in selected if p[0] < int(args.pair_end)]
    if args.pair_shard_count is not None:
        count = int(args.pair_shard_count)
        index = int(args.pair_shard_index)
        if index < 0 or index >= count:
            raise ValueError("--pair-shard-index must satisfy 0 <= index < count")
        selected = [p for p in selected if p[0] % count == index]
    if args.limit_pairs is not None:
        selected = selected[: int(args.limit_pairs)]
    return selected


def write_static_outputs(
    out_root: Path,
    specs: list[Any],
    alphas: np.ndarray,
    bbox_records: dict[str, dict[str, Any]],
    lyap_cfg: LyapConfig,
    cae_config: CaeConfig,
    transfer_config: TransferConfig,
    args: argparse.Namespace,
    selected_pairs: list[tuple[int, str, str]],
) -> None:
    save_json(out_root / "estimated_grid_limits.json", bbox_records)
    save_json(
        out_root / "config.json",
        {
            "experiment": "classification20_pairwise_cae",
            "systems": [
                {"key": spec.key, "label": spec.label, "source": spec.source}
                for spec in specs
            ],
            "target_grid_limits": TARGET_LIMITS,
            "target_velocity": TARGET_VELOCITY,
            "alphas": alphas.tolist(),
            "init_point": list(DEFAULT_INIT_POINT),
            "lyap_config": asdict(lyap_cfg),
            "cae_config": asdict(cae_config),
            "transfer_config": asdict(transfer_config),
            "feature_layers": "CAE encoder feature_layers only, all three layers",
            "style_loss": "disabled",
            "content_weights": "per layer: A=(1-alpha)/3, B=alpha/3",
            "fusion_initialization": "linear homotopy tensor",
            "closed_form_projection": "vec2ode linear regression, max_order=2",
            "num_pairs_total": len(list(combinations([spec.key for spec in specs], 2))),
            "num_pairs_selected_this_run": len(selected_pairs),
            "num_eval_records_expected_total": len(
                list(combinations([spec.key for spec in specs], 2))
            )
            * len(alphas),
            "num_interior_fusions_expected_total": len(
                list(combinations([spec.key for spec in specs], 2))
            )
            * max(0, len(alphas) - 2),
            "homotopy_results_dir": str(args.homotopy_results_dir),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "distance_components": [
                "endpoint_gap",
                "barrier",
                "roughness",
                "chaos_loss",
            ],
            "distance_normalization": "percentile_rank_across_pairs",
            "sharding": {
                "pair_shard_index": args.pair_shard_index,
                "pair_shard_count": args.pair_shard_count,
                "pair_start": args.pair_start,
                "pair_end": args.pair_end,
                "limit_pairs": args.limit_pairs,
            },
        },
    )


def merge_outputs(
    out_root: Path,
    specs: list[Any],
    alphas: np.ndarray,
    allow_partial: bool,
    logger: logging.Logger,
) -> None:
    names = [spec.key for spec in specs]
    pair_names = list(combinations(names, 2))
    expected_eval = len(pair_names) * len(alphas)

    eval_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    cae_rows: dict[str, dict[str, Any]] = {}
    fusion_rows: dict[tuple[str, int], dict[str, Any]] = {}
    shard_dirs = sorted((out_root / "shards").glob("shard_*"))
    single_dir = out_root / "shards" / "single"
    if single_dir.exists():
        shard_dirs.append(single_dir)

    for shard_dir in shard_dirs:
        for row in read_csv_rows(shard_dir / "eval_records.csv"):
            parsed = parse_eval_row(row)
            eval_by_key[(str(parsed["pair_id"]), int(parsed["alpha_index"]))] = parsed
        for row in read_csv_rows(shard_dir / "cae_records.csv"):
            cae_rows[row["pair_id"]] = row
        for row in read_csv_rows(shard_dir / "fusion_records.csv"):
            fusion_rows[(row["pair_id"], int(row["alpha_index"]))] = row

    eval_rows = list(eval_by_key.values())
    eval_rows.sort(
        key=lambda r: (
            int(r.get("pair_index", 10**9)),
            str(r["pair_id"]),
            int(r["alpha_index"]),
        )
    )
    if len(eval_rows) != expected_eval and not allow_partial:
        raise RuntimeError(
            f"expected {expected_eval} eval rows, found {len(eval_rows)}"
        )
    write_records_csv(out_root / "eval_records.csv", eval_rows)

    write_csv(
        out_root / "cae_records.csv",
        sorted(cae_rows.values(), key=lambda r: int(r["pair_index"])),
        CAE_FIELDNAMES,
    )
    write_csv(
        out_root / "fusion_records.csv",
        sorted(
            fusion_rows.values(),
            key=lambda r: (int(r["pair_index"]), int(r["alpha_index"])),
        ),
        FUSION_FIELDNAMES,
    )

    spectra_by_pair = spectra_from_records(eval_rows, pair_names, alphas)
    component_rows = build_pair_component_rows(pair_names, alphas, spectra_by_pair)
    write_pair_components(out_root / "pair_components.csv", component_rows)

    distance_matrix = matrix_from_pair_rows(names, component_rows, "distance")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise AssertionError("distance matrix is not symmetric")
    if not np.allclose(np.diag(distance_matrix), 0.0):
        raise AssertionError("distance matrix diagonal is not zero")
    write_matrix_csv(out_root / "distance_matrix.csv", names, distance_matrix)

    component_dir = out_root / "component_matrices"
    component_dir.mkdir(parents=True, exist_ok=True)
    for component in ["endpoint_gap", "barrier", "roughness", "chaos_loss"]:
        write_matrix_csv(
            component_dir / f"{component}.csv",
            names,
            matrix_from_pair_rows(names, component_rows, component),
        )
        write_matrix_csv(
            component_dir / f"{component}_rank.csv",
            names,
            matrix_from_pair_rows(names, component_rows, f"{component}_rank"),
        )
    write_nearest_neighbors(out_root / "nearest_neighbors.csv", names, distance_matrix)

    coef_bundle: dict[str, np.ndarray] = {}
    for path in sorted((out_root / PAIR_COEF_DIRNAME).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                coef_bundle[f"{path.stem}__{key}"] = np.array(data[key], copy=True)
    if coef_bundle:
        np.savez_compressed(out_root / "fused_coefficients.npz", **coef_bundle)

    n_fail = sum(1 for row in eval_rows if row["status"] != "OK")
    logger.info("Merged eval records: %d/%d", len(eval_rows), expected_eval)
    logger.info("Merged CAE records: %d", len(cae_rows))
    logger.info("Merged fusion records: %d", len(fusion_rows))
    logger.info("Failed Lyapunov evaluations: %d/%d", n_fail, len(eval_rows))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="20-system pair-specific CAE classification."
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--homotopy-results-dir", type=Path, default=DEFAULT_HOMOTOPY_RESULTS_DIR
    )

    return parser.parse_args(argv)


def configs_from_args(args):
    lyap_cfg = LyapConfig(
        h=float(args.lyap_h),
        burn_steps=int(args.burn_steps),
        lyap_steps=int(args.lyap_steps),
        max_abs=float(args.max_abs),
    )
    cae_config = CaeConfig(
        grid_shape=parse_int_tuple(args.grid_shape),
        depth=int(args.cae_depth),
        batch_size=int(args.cae_batch_size),
        max_batches=int(args.cae_max_batches),
        lr=float(args.cae_lr),
        turb_scale=float(args.cae_turb_scale),
        reduce_factor=float(args.cae_reduce_factor),
        reduce_patience=int(args.cae_reduce_patience),
        early_lr=float(args.cae_early_lr),
        log_every=int(args.cae_log_every),
    )
    transfer_config = TransferConfig(
        max_iters=int(args.transfer_max_iters),
        lr=float(args.transfer_lr),
        reduce_factor=float(args.transfer_reduce_factor),
        reduce_patience=int(args.transfer_reduce_patience),
        early_lr=float(args.transfer_early_lr),
        log_every=int(args.transfer_log_every),
    )

    return lyap_cfg, cae_config, transfer_config


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.pair_shard_count is None and args.pair_shard_index is not None:
        raise ValueError("--pair-shard-index requires --pair-shard-count")
    if args.pair_shard_count is not None and args.pair_shard_index is None:
        raise ValueError("--pair-shard-count requires --pair-shard-index")

    if args.smoke_test:
        out_root = args.out_root if args.out_root is not None else SMOKE_RESULTS_DIR
        limit_systems = args.limit_systems if args.limit_systems is not None else 2
        alphas = (
            parse_alpha_list(args.alphas)
            if args.alphas is not None
            else np.array([0.0, 0.5, 1.0])
        )
        if args.limit_pairs is None:
            args.limit_pairs = 1
    else:
        out_root = args.out_root if args.out_root is not None else DEFAULT_RESULTS_DIR
        limit_systems = args.limit_systems
        alphas = (
            parse_alpha_list(args.alphas)
            if args.alphas is not None
            else DEFAULT_ALPHAS.copy()
        )

    out_root = resolve_inside_project(out_root)
    args.homotopy_results_dir = resolve_inside_project(args.homotopy_results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / PAIR_COEF_DIRNAME).mkdir(parents=True, exist_ok=True)
    (out_root / "pair_models").mkdir(parents=True, exist_ok=True)
    (out_root / "component_matrices").mkdir(parents=True, exist_ok=True)

    this_shard = shard_name(args)
    log_path = out_root / "logs" / f"{this_shard}.log"
    if args.merge_only:
        log_path = out_root / "run.log"
    logger = setup_logger(log_path, name=f"content_only_paircae_{this_shard}")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    lyap_cfg, cae_config, transfer_config = configs_from_args(args)

    specs = build_system_specs()
    if limit_systems is not None:
        specs = specs[: int(limit_systems)]
    if len(specs) < 2:
        raise ValueError("need at least two systems")
    names = [spec.key for spec in specs]
    pairs = all_pairs(names)
    selected_pairs = select_pairs_for_args(pairs, args)

    bbox_records_raw = load_homotopy_bbox(args.homotopy_results_dir)
    normalized, bbox_records = normalize_systems_from_bbox(
        specs, bbox_records_raw, lyap_cfg.h
    )
    grids = calc_grids(TARGET_LIMITS, list(cae_config.grid_shape))

    if args.merge_only:
        merge_outputs(out_root, specs, alphas, bool(args.allow_partial), logger)
        return

    if args.pair_shard_index in (None, 0):
        write_static_outputs(
            out_root,
            specs,
            alphas,
            bbox_records,
            lyap_cfg,
            cae_config,
            transfer_config,
            args,
            selected_pairs,
        )

    logger.info("Systems: %s", names)
    logger.info("Alpha grid: %s", alphas.tolist())
    logger.info("Selected pairs for %s: %d", this_shard, len(selected_pairs))
    logger.info("Device: %s", device)
    logger.info("Lyapunov config: %s", asdict(lyap_cfg))
    logger.info("CAE config: %s", asdict(cae_config))
    logger.info("Transfer config: %s", asdict(transfer_config))

    shard_dir = out_root / "shards" / this_shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_eval_path = shard_dir / "eval_records.csv"
    shard_cae_path = shard_dir / "cae_records.csv"
    shard_fusion_path = shard_dir / "fusion_records.csv"

    existing_eval = load_shard_eval_map(shard_eval_path) if args.resume else {}
    cae_rows = read_csv_rows(shard_cae_path) if args.resume else []
    fusion_rows = read_csv_rows(shard_fusion_path) if args.resume else []

    for local_i, (p_index, name_a, name_b) in enumerate(selected_pairs, start=1):
        logger.info(
            "Start pair %d/%d: %03d %s",
            local_i,
            len(selected_pairs),
            p_index,
            pair_id(name_a, name_b),
        )
        try:
            cae_row, new_fusion_rows, new_eval_rows = run_one_pair(
                p_index,
                name_a,
                name_b,
                normalized,
                grids,
                alphas,
                cae_config,
                transfer_config,
                lyap_cfg,
                out_root,
                existing_eval,
                bool(args.resume),
                device,
                int(args.seed),
                logger,
            )
            if cae_row:
                cae_rows = [
                    row for row in cae_rows if row.get("pair_id") != cae_row["pair_id"]
                ]
                cae_rows.append(cae_row)
            for row in new_fusion_rows:
                fusion_rows = [
                    old
                    for old in fusion_rows
                    if not (
                        old.get("pair_id") == row["pair_id"]
                        and int(old.get("alpha_index", -1)) == int(row["alpha_index"])
                    )
                ]
                fusion_rows.append(row)
            for row in new_eval_rows:
                existing_eval[(str(row["pair_id"]), int(row["alpha_index"]))] = row

            write_csv(
                shard_cae_path,
                sorted(cae_rows, key=lambda r: int(r["pair_index"])),
                CAE_FIELDNAMES,
            )
            write_csv(
                shard_fusion_path,
                sorted(
                    fusion_rows,
                    key=lambda r: (int(r["pair_index"]), int(r["alpha_index"])),
                ),
                FUSION_FIELDNAMES,
            )
            write_csv(
                shard_eval_path,
                sorted(
                    existing_eval.values(),
                    key=lambda r: (
                        int(r.get("pair_index", p_index)),
                        str(r["pair_id"]),
                        int(r["alpha_index"]),
                    ),
                ),
                EVAL_FIELDNAMES_SHARD,
            )
            logger.info("Finished pair %03d %s", p_index, pair_id(name_a, name_b))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Pair %03d %s failed: %s", p_index, pair_id(name_a, name_b), exc
            )
            fail_row = {
                "pair_index": p_index,
                "pair_id": pair_id(name_a, name_b),
                "system_a": name_a,
                "system_b": name_b,
                "status": "FAIL",
                "train_batches": 0,
                "best_loss": np.nan,
                "final_loss": np.nan,
                "final_lr": np.nan,
                "recon_mse_a": np.nan,
                "recon_mse_b": np.nan,
                "model_path": "",
                "elapsed_sec": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            cae_rows = [
                row for row in cae_rows if row.get("pair_id") != fail_row["pair_id"]
            ]
            cae_rows.append(fail_row)
            write_csv(
                shard_cae_path,
                sorted(cae_rows, key=lambda r: int(r["pair_index"])),
                CAE_FIELDNAMES,
            )

    logger.info("Shard %s done.", this_shard)


if __name__ == "__main__":
    main()
