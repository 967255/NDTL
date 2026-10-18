from __future__ import annotations

from .arguments import add_shared_arguments

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ndtl.paths import ROOT as PROJECT_ROOT

import experiments.classification.train20 as base  # noqa: E402
from experiments.classification.dysts import (  # noqa: E402
    FLOWS_PATH,
    METADATA_PATH,
    build_dysts_quadratic_specs,
)
from experiments.classification.systems import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_INIT_POINT,
    TARGET_LIMITS,
    TARGET_VELOCITY,
)
from ndtl.metrics import save_json  # noqa: E402
from ndtl.fields import calc_grids  # noqa: E402

DEFAULT_RESULTS_DIR = Path("outputs/classification/systems52")
SMOKE_RESULTS_DIR = Path("outputs/classification/systems52_smoke")
DEFAULT_REFERENCE_RESULTS_DIR = Path("experiments/classification/results/systems20")
DEFAULT_DYSTS_HOMOTOPY_DIR = Path("experiments/classification/configs/dysts51")
NEWTON_DYSTS_KEY = "newton_liepnik"
NEWTON_CANONICAL_KEY = "newton_leipnik"


def canonical_key(key: str) -> str:
    if key == NEWTON_DYSTS_KEY:
        return NEWTON_CANONICAL_KEY
    return key


def build_final_specs(
    limit_systems: int | None = None,
) -> tuple[list[Any], list[str], list[str], list[dict[str, Any]]]:
    old_specs = base.build_system_specs()
    old_names = [spec.key for spec in old_specs]
    old_key_set = set(old_names)

    dysts_specs, selection_rows = build_dysts_quadratic_specs(FLOWS_PATH, METADATA_PATH)
    new_specs = []
    for spec in dysts_specs:
        if canonical_key(spec.key) in old_key_set:
            continue
        new_specs.append(spec)

    specs = old_specs + new_specs
    if limit_systems is not None:
        specs = specs[: int(limit_systems)]
        old_names = [name for name in old_names if name in {spec.key for spec in specs}]
        new_names = [spec.key for spec in specs if spec.key not in set(old_names)]
    else:
        new_names = [spec.key for spec in new_specs]

    return specs, old_names, new_names, selection_rows


def build_pair_list(
    old_names: list[str], new_names: list[str]
) -> list[tuple[int, str, str]]:
    pairs: list[tuple[int, str, str]] = []
    for name_a, name_b in combinations(old_names, 2):
        pairs.append((len(pairs), name_a, name_b))
    for name_a in old_names:
        for name_b in new_names:
            pairs.append((len(pairs), name_a, name_b))
    for name_a, name_b in combinations(new_names, 2):
        pairs.append((len(pairs), name_a, name_b))
    return pairs


def pair_names_from_indexed(pairs: list[tuple[int, str, str]]) -> list[tuple[str, str]]:
    return [(name_a, name_b) for _, name_a, name_b in pairs]


def combine_bbox_records(
    specs: list[Any],
    old_names: list[str],
    reference_content_dir: Path,
    dysts_homotopy_dir: Path,
) -> dict[str, dict[str, Any]]:
    old_bbox = base.load_homotopy_bbox(reference_content_dir)
    dysts_bbox = base.load_homotopy_bbox(dysts_homotopy_dir)
    old_set = set(old_names)
    records: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.key in old_set:
            if spec.key not in old_bbox:
                raise KeyError(f"missing reference bbox for {spec.key}")
            records[spec.key] = dict(old_bbox[spec.key])
            records[spec.key]["bbox_source"] = str(reference_content_dir)
        else:
            source_key = (
                NEWTON_DYSTS_KEY if spec.key == NEWTON_CANONICAL_KEY else spec.key
            )
            if source_key not in dysts_bbox:
                raise KeyError(
                    f"missing DYSTS homotopy bbox for {spec.key} (source key {source_key})"
                )
            records[spec.key] = dict(dysts_bbox[source_key])
            records[spec.key]["bbox_source"] = str(dysts_homotopy_dir)
    return records


def write_final_systems_csv(path: Path, specs: list[Any], old_names: list[str]) -> None:
    old_set = set(old_names)
    fieldnames = ["order", "key", "label", "source", "role"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for order, spec in enumerate(specs):
            writer.writerow(
                {
                    "order": order,
                    "key": spec.key,
                    "label": spec.label,
                    "source": spec.source,
                    "role": (
                        "reference_20_reused" if spec.key in old_set else "dysts_new"
                    ),
                }
            )


def filter_work_pairs(
    pairs: list[tuple[int, str, str]],
    old_names: list[str],
    skip_reference_block: bool,
) -> list[tuple[int, str, str]]:
    if not skip_reference_block:
        return pairs
    old_set = set(old_names)
    return [(idx, a, b) for idx, a, b in pairs if not (a in old_set and b in old_set)]


def pair_index_map(pairs: list[tuple[int, str, str]]) -> dict[str, int]:
    return {base.pair_id(name_a, name_b): idx for idx, name_a, name_b in pairs}


def read_reference_eval_records(
    reference_content_dir: Path, valid_pair_ids: set[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in base.read_csv_rows(reference_content_dir / "eval_records.csv"):
        parsed = base.parse_eval_row(row)
        pid = str(parsed["pair_id"])
        if pid in valid_pair_ids:
            out[(pid, int(parsed["alpha_index"]))] = parsed
    return out


def read_shard_dirs(out_root: Path) -> list[Path]:
    shard_dirs = sorted((out_root / "shards").glob("shard_*"))
    single_dir = out_root / "shards" / "single"
    if single_dir.exists():
        shard_dirs.append(single_dir)
    return shard_dirs


def load_merged_eval_records(
    out_root: Path,
    reference_content_dir: Path,
    reference_pair_ids: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    eval_by_key = read_reference_eval_records(reference_content_dir, reference_pair_ids)
    for shard_dir in read_shard_dirs(out_root):
        for row in base.read_csv_rows(shard_dir / "eval_records.csv"):
            parsed = base.parse_eval_row(row)
            eval_by_key[(str(parsed["pair_id"]), int(parsed["alpha_index"]))] = parsed
    return eval_by_key


def normalize_row_pair_index(
    row: dict[str, Any], index_by_pair: dict[str, int]
) -> dict[str, Any]:
    out = dict(row)
    pid = str(out["pair_id"])
    if pid in index_by_pair:
        out["pair_index"] = index_by_pair[pid]
    return out


def merge_outputs52(
    out_root: Path,
    reference_content_dir: Path,
    specs: list[Any],
    pairs: list[tuple[int, str, str]],
    old_names: list[str],
    alphas: np.ndarray,
    allow_partial: bool,
    logger: Any,
) -> None:
    names = [spec.key for spec in specs]
    pair_names = pair_names_from_indexed(pairs)
    index_by_pair = pair_index_map(pairs)
    old_set = set(old_names)
    reference_pair_ids = {
        base.pair_id(name_a, name_b)
        for _, name_a, name_b in pairs
        if name_a in old_set and name_b in old_set
    }
    expected_eval = len(pair_names) * len(alphas)

    eval_by_key = load_merged_eval_records(
        out_root, reference_content_dir, reference_pair_ids
    )
    eval_rows = [
        normalize_row_pair_index(row, index_by_pair) for row in eval_by_key.values()
    ]
    eval_rows.sort(
        key=lambda r: (int(r.get("pair_index", 10**9)), int(r["alpha_index"]))
    )
    if len(eval_rows) != expected_eval and not allow_partial:
        raise RuntimeError(
            f"expected {expected_eval} eval rows, found {len(eval_rows)}"
        )
    base.write_records_csv(out_root / "eval_records.csv", eval_rows)

    cae_rows: dict[str, dict[str, Any]] = {}
    for row in base.read_csv_rows(reference_content_dir / "cae_records.csv"):
        if row.get("pair_id") in reference_pair_ids:
            cae_rows[row["pair_id"]] = normalize_row_pair_index(row, index_by_pair)
    fusion_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in base.read_csv_rows(reference_content_dir / "fusion_records.csv"):
        if row.get("pair_id") in reference_pair_ids:
            row_norm = normalize_row_pair_index(row, index_by_pair)
            fusion_rows[(row["pair_id"], int(row["alpha_index"]))] = row_norm

    for shard_dir in read_shard_dirs(out_root):
        for row in base.read_csv_rows(shard_dir / "cae_records.csv"):
            cae_rows[row["pair_id"]] = normalize_row_pair_index(row, index_by_pair)
        for row in base.read_csv_rows(shard_dir / "fusion_records.csv"):
            row_norm = normalize_row_pair_index(row, index_by_pair)
            fusion_rows[(row["pair_id"], int(row["alpha_index"]))] = row_norm

    base.write_csv(
        out_root / "cae_records.csv",
        sorted(cae_rows.values(), key=lambda r: int(r["pair_index"])),
        base.CAE_FIELDNAMES,
    )
    base.write_csv(
        out_root / "fusion_records.csv",
        sorted(
            fusion_rows.values(),
            key=lambda r: (int(r["pair_index"]), int(r["alpha_index"])),
        ),
        base.FUSION_FIELDNAMES,
    )

    spectra_by_pair = base.spectra_from_records(eval_rows, pair_names, alphas)
    component_rows = base.build_pair_component_rows(pair_names, alphas, spectra_by_pair)
    base.write_pair_components(out_root / "pair_components.csv", component_rows)

    distance_matrix = base.matrix_from_pair_rows(names, component_rows, "distance")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise AssertionError("distance matrix is not symmetric")
    if not np.allclose(np.diag(distance_matrix), 0.0):
        raise AssertionError("distance matrix diagonal is not zero")
    base.write_matrix_csv(out_root / "distance_matrix.csv", names, distance_matrix)

    component_dir = out_root / "component_matrices"
    component_dir.mkdir(parents=True, exist_ok=True)
    for component in ["endpoint_gap", "barrier", "roughness", "chaos_loss"]:
        base.write_matrix_csv(
            component_dir / f"{component}.csv",
            names,
            base.matrix_from_pair_rows(names, component_rows, component),
        )
        base.write_matrix_csv(
            component_dir / f"{component}_rank.csv",
            names,
            base.matrix_from_pair_rows(names, component_rows, f"{component}_rank"),
        )
    base.write_nearest_neighbors(
        out_root / "nearest_neighbors.csv", names, distance_matrix
    )

    coef_bundle: dict[str, np.ndarray] = {}
    for source_root in [reference_content_dir, out_root]:
        for path in sorted((source_root / base.PAIR_COEF_DIRNAME).glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    coef_bundle[f"{path.stem}__{key}"] = np.array(data[key], copy=True)
    if coef_bundle:
        np.savez_compressed(out_root / "fused_coefficients.npz", **coef_bundle)

    n_fail = sum(1 for row in eval_rows if row["status"] != "OK")
    logger.info("Merged eval records: %d/%d", len(eval_rows), expected_eval)
    logger.info("Merged CAE records: %d/%d", len(cae_rows), len(pair_names))
    logger.info("Merged fusion records: %d", len(fusion_rows))
    logger.info("Failed Lyapunov evaluations: %d/%d", n_fail, len(eval_rows))


def write_static_outputs52(
    out_root: Path,
    specs: list[Any],
    old_names: list[str],
    new_names: list[str],
    pairs: list[tuple[int, str, str]],
    work_pairs: list[tuple[int, str, str]],
    selected_pairs: list[tuple[int, str, str]],
    alphas: np.ndarray,
    bbox_records: dict[str, dict[str, Any]],
    lyap_cfg: Any,
    cae_config: Any,
    transfer_config: Any,
    args: argparse.Namespace,
) -> None:
    save_json(out_root / "estimated_grid_limits.json", bbox_records)
    write_final_systems_csv(out_root / "selected_systems.csv", specs, old_names)
    save_json(
        out_root / "config.json",
        {
            "experiment": "content_only_pair_specific_cae_dysts52",
            "systems": [
                {"key": spec.key, "label": spec.label, "source": spec.source}
                for spec in specs
            ],
            "reference_20_systems": old_names,
            "new_dysts_systems": new_names,
            "newton_key_policy": f"{NEWTON_DYSTS_KEY} treated as {NEWTON_CANONICAL_KEY}",
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
            "num_systems": len(specs),
            "num_pairs_total": len(pairs),
            "num_pairs_reused_from_reference_20": len(pairs) - len(work_pairs),
            "num_pairs_to_compute": len(work_pairs),
            "num_pairs_selected_this_run": len(selected_pairs),
            "num_eval_records_expected_total": len(pairs) * len(alphas),
            "num_eval_records_to_compute": len(work_pairs) * len(alphas),
            "num_interior_fusions_to_compute": len(work_pairs)
            * max(0, len(alphas) - 2),
            "reference_results_dir": str(args.reference_results_dir),
            "dysts_homotopy_results_dir": str(args.dysts_homotopy_results_dir),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "distance_components": [
                "endpoint_gap",
                "barrier",
                "roughness",
                "chaos_loss",
            ],
            "distance_normalization": "percentile_rank_across_pairs",
            "pair_order": "reference_20_internal, reference_20_to_new_dysts, new_dysts_internal",
            "sharding": {
                "pair_shard_index": args.pair_shard_index,
                "pair_shard_count": args.pair_shard_count,
                "pair_start": args.pair_start,
                "pair_end": args.pair_end,
                "limit_pairs": args.limit_pairs,
            },
        },
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="52-system pair-specific CAE content-only fusion benchmark."
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--reference-results-dir", type=Path, default=DEFAULT_REFERENCE_RESULTS_DIR
    )
    parser.add_argument(
        "--dysts-homotopy-results-dir", type=Path, default=DEFAULT_DYSTS_HOMOTOPY_DIR
    )
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.pair_shard_count is None and args.pair_shard_index is not None:
        raise ValueError("--pair-shard-index requires --pair-shard-count")
    if args.pair_shard_count is not None and args.pair_shard_index is None:
        raise ValueError("--pair-shard-count requires --pair-shard-index")

    if args.smoke_test:
        out_root = args.out_root if args.out_root is not None else SMOKE_RESULTS_DIR
        alphas = (
            base.parse_alpha_list(args.alphas)
            if args.alphas is not None
            else np.array([0.0, 0.5, 1.0])
        )
        if args.limit_pairs is None:
            args.limit_pairs = 3
    else:
        out_root = args.out_root if args.out_root is not None else DEFAULT_RESULTS_DIR
        alphas = (
            base.parse_alpha_list(args.alphas)
            if args.alphas is not None
            else DEFAULT_ALPHAS.copy()
        )

    out_root = base.resolve_inside_project(out_root)
    args.reference_results_dir = base.resolve_inside_project(args.reference_results_dir)
    args.dysts_homotopy_results_dir = base.resolve_inside_project(
        args.dysts_homotopy_results_dir
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / base.PAIR_COEF_DIRNAME).mkdir(parents=True, exist_ok=True)
    (out_root / "pair_models").mkdir(parents=True, exist_ok=True)
    (out_root / "component_matrices").mkdir(parents=True, exist_ok=True)

    this_shard = base.shard_name(args)
    log_path = out_root / "logs" / f"{this_shard}.log"
    if args.merge_only or args.dry_run:
        log_path = out_root / "run.log"
    logger = base.setup_logger(
        log_path, name=f"content_only_paircae_dysts52_{this_shard}"
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(int(args.seed))

    lyap_cfg, cae_config, transfer_config = base.configs_from_args(args)

    specs, old_names, new_names, _ = build_final_specs(args.limit_systems)
    if len(specs) != 52 and args.limit_systems is None:
        raise RuntimeError(f"expected 52 systems, got {len(specs)}")
    if len(old_names) != 20 and args.limit_systems is None:
        raise RuntimeError(f"expected 20 reference systems, got {len(old_names)}")
    names = [spec.key for spec in specs]
    pairs = build_pair_list(old_names, new_names)
    work_pairs = filter_work_pairs(pairs, old_names, skip_reference_block=True)
    selected_pairs = base.select_pairs_for_args(work_pairs, args)

    logger.info("Systems total: %d", len(specs))
    logger.info("Reference systems reused: %d", len(old_names))
    logger.info("New DYSTS systems: %d", len(new_names))
    logger.info("Pairs total: %d", len(pairs))
    logger.info("Reference 20x20 pairs skipped: %d", len(pairs) - len(work_pairs))
    logger.info("Pairs to compute: %d", len(work_pairs))
    logger.info("Selected pairs for %s: %d", this_shard, len(selected_pairs))
    logger.info("System order: %s", names)

    if args.dry_run:
        expected_new_eval = len(work_pairs) * len(alphas)
        expected_total_eval = len(pairs) * len(alphas)
        print(
            json.dumps(
                {
                    "num_systems": len(specs),
                    "num_pairs_total": len(pairs),
                    "num_pairs_reused_from_reference_20": len(pairs) - len(work_pairs),
                    "num_pairs_to_compute": len(work_pairs),
                    "num_eval_records_to_compute": expected_new_eval,
                    "num_eval_records_total_after_merge": expected_total_eval,
                    "num_selected_pairs_this_run": len(selected_pairs),
                    "first_selected_pairs": [
                        {"pair_index": idx, "system_a": a, "system_b": b}
                        for idx, a, b in selected_pairs[:10]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.merge_only:
        merge_outputs52(
            out_root,
            args.reference_results_dir,
            specs,
            pairs,
            old_names,
            alphas,
            bool(args.allow_partial),
            logger,
        )
        return

    bbox_records_raw = combine_bbox_records(
        specs,
        old_names,
        args.reference_results_dir,
        args.dysts_homotopy_results_dir,
    )
    normalized, bbox_records = base.normalize_systems_from_bbox(
        specs, bbox_records_raw, lyap_cfg.h
    )
    grids = calc_grids(TARGET_LIMITS, list(cae_config.grid_shape))
    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    if args.pair_shard_index in (None, 0):
        write_static_outputs52(
            out_root,
            specs,
            old_names,
            new_names,
            pairs,
            work_pairs,
            selected_pairs,
            alphas,
            bbox_records,
            lyap_cfg,
            cae_config,
            transfer_config,
            args,
        )

    logger.info("Alpha grid: %s", alphas.tolist())
    logger.info("Device: %s", device)
    logger.info("Lyapunov config: %s", asdict(lyap_cfg))
    logger.info("CAE config: %s", asdict(cae_config))
    logger.info("Transfer config: %s", asdict(transfer_config))

    shard_dir = out_root / "shards" / this_shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_eval_path = shard_dir / "eval_records.csv"
    shard_cae_path = shard_dir / "cae_records.csv"
    shard_fusion_path = shard_dir / "fusion_records.csv"

    existing_eval = base.load_shard_eval_map(shard_eval_path) if args.resume else {}
    cae_rows = base.read_csv_rows(shard_cae_path) if args.resume else []
    fusion_rows = base.read_csv_rows(shard_fusion_path) if args.resume else []

    for local_i, (p_index, name_a, name_b) in enumerate(selected_pairs, start=1):
        logger.info(
            "Start pair %d/%d: %03d %s",
            local_i,
            len(selected_pairs),
            p_index,
            base.pair_id(name_a, name_b),
        )
        try:
            cae_row, new_fusion_rows, new_eval_rows = base.run_one_pair(
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

            base.write_csv(
                shard_cae_path,
                sorted(cae_rows, key=lambda r: int(r["pair_index"])),
                base.CAE_FIELDNAMES,
            )
            base.write_csv(
                shard_fusion_path,
                sorted(
                    fusion_rows,
                    key=lambda r: (int(r["pair_index"]), int(r["alpha_index"])),
                ),
                base.FUSION_FIELDNAMES,
            )
            base.write_csv(
                shard_eval_path,
                sorted(
                    existing_eval.values(),
                    key=lambda r: (
                        int(r.get("pair_index", p_index)),
                        str(r["pair_id"]),
                        int(r["alpha_index"]),
                    ),
                ),
                base.EVAL_FIELDNAMES_SHARD,
            )
            logger.info("Finished pair %03d %s", p_index, base.pair_id(name_a, name_b))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Pair %03d %s failed: %s", p_index, base.pair_id(name_a, name_b), exc
            )
            fail_row = {
                "pair_index": p_index,
                "pair_id": base.pair_id(name_a, name_b),
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
            base.write_csv(
                shard_cae_path,
                sorted(cae_rows, key=lambda r: int(r["pair_index"])),
                base.CAE_FIELDNAMES,
            )

    logger.info("Shard %s done.", this_shard)


if __name__ == "__main__":
    main()
