import json, os, logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from ndtl.models import CAE
from ndtl.sampling import DataSampler
from ndtl.fields import calc_grids
from ndtl.metrics import (
    FeatureConfig,
    compute_system_features,
    summarize_over_inits,
    save_json,
)
from ndtl.fields import vec2ode
import argparse
from pathlib import Path
from ndtl.paths import ROOT


def main():
    _parser = argparse.ArgumentParser(
        description="Evaluate a classic fusion array with an explicit publication preset."
    )
    _parser.add_argument("--experiment", default="E01")
    _parser.add_argument("--vec-path")
    _parser.add_argument("--output-dir", required=True)
    _args = _parser.parse_args()
    _root = ROOT
    _catalog = json.loads(
        (_root / "experiments/classic/configs/evaluation.json").read_text()
    )
    if _args.experiment not in _catalog:
        _parser.error("unknown experiment preset")
    _record_dir = _root / _catalog[_args.experiment]["evaluation_results"][0]
    _record = json.loads((_record_dir / "vec_0/config.json").read_text())
    VEC_PATH = _args.vec_path or str(
        _root / _catalog[_args.experiment]["source_arrays"][0]
    )
    CAE_INFO = _record["CAE_INFO"]
    META_INFO = _record["META_INFO"]
    grid_shape = _record["grid_shape"]
    grid_limits = _record["grid_limits"]
    INIT_POINTS = _record["init_points"]
    PIC_ROOT_PATH = _args.output_dir
    os.makedirs(PIC_ROOT_PATH, exist_ok=True)
    logger = logging.getLogger("classic_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    _handler = logging.FileHandler(
        os.path.join(PIC_ROOT_PATH, "info.log"), encoding="utf-8"
    )
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.addHandler(logging.StreamHandler())
    FEATURE_CFG = FeatureConfig(
        traj_length=5000,
        omit_points=0,
        h=None,
        burn_in=500,
        stride=5,
        twonn_approximate=False,
        twonn_theiler=10,
        twonn_k_search=64,
        lyap_tol=1e-08,
        lyap_min_tpts=10,
        lyap_fd_eps=None,
        save_traj_npz=True,
        save_lyap_npz=True,
        save_plots=True,
    )
    "BODY"
    vecs = np.load(VEC_PATH)
    logger.info(f"vecs.shape = {vecs.shape}")
    grids = calc_grids(grid_limits=grid_limits, grid_shape=grid_shape)
    num_vecs = vecs.shape[0]
    logger.info(f"num_vecs = {num_vecs}")
    K = len(INIT_POINTS)
    D = 3
    lyap_spectra = np.full((num_vecs, K, D), np.nan, dtype=float)
    sim_status = np.empty((num_vecs, K), dtype=object)
    intrinsic_dims = np.full((num_vecs, K), np.nan, dtype=float)
    kaplan_yorke_dims = np.full((num_vecs, K), np.nan, dtype=float)
    alphas = np.asarray(_catalog[_args.experiment]["fusion_weights"], dtype=float)
    if len(alphas) != num_vecs:
        raise ValueError("field count does not match the selected experiment weights")
    for vec_index in range(num_vecs):
        logger.info(f"=== start vec index {vec_index} ===")
        vec = vecs[vec_index]
        logger.info(f"vec[{vec_index}].shape = {vec.shape}")
        vec_dir = os.path.join(PIC_ROOT_PATH, f"vec_{vec_index}")
        os.makedirs(vec_dir, exist_ok=True)
        ode_hat = vec2ode(
            vec,
            grids,
            sparse_coef=None,
            max_order=2,
            extra_term_dicts=None,
            mode="linear",
            alpha=0.1,
            alphas=None,
            cv=5,
            fit_intercept=True,
            name=f"ode_hat_{vec_index}",
            verbose=False,
            return_diagnostics=False,
            grid_limits=None,
        )
        config = {
            "VEC_PATH": VEC_PATH,
            "CAE_INFO": CAE_INFO,
            "META_INFO": META_INFO,
            "grid_shape": grid_shape,
            "grid_limits": grid_limits,
            "vec_index": vec_index,
            "init_points": INIT_POINTS,
        }
        config_path = os.path.join(vec_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"[vec {vec_index}] config saved to {config_path}")
        info = ode_hat.get_info()
        logger.info(f"[vec {vec_index}] ode_hat info: {info}")
        info_path = os.path.join(vec_dir, "ode_info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(str(info))
        logger.info(f"[vec {vec_index}] ode info saved to {info_path}")
        per_init_results = []
        for k, init_point in enumerate(INIT_POINTS):
            init_dir = os.path.join(vec_dir, f"init_{k}")
            os.makedirs(init_dir, exist_ok=True)
            ode_hat.plot(
                pic_save_dir=os.path.join(init_dir, "trajectory"), init_point=init_point
            )
            logger.info(
                f"[vec {vec_index}] trajectory for init_{k}={init_point} saved under {init_dir}"
            )
            features_dir = os.path.join(init_dir, "features")
            try:
                res = compute_system_features(
                    ode=ode_hat,
                    init_point=np.array(init_point, dtype=float),
                    out_dir=features_dir,
                    cfg=FEATURE_CFG,
                )
                per_init_results.append(res)
                logger.info(
                    f"[vec {vec_index}] features done for init_{k}={init_point}: ID={res['intrinsic_dimension_twonn']:.4f}, KY={res['kaplan_yorke_dimension']:.4f}, spectrum={np.array(res['lyapunov_spectrum'])}"
                )
                spec = np.asarray(res["lyapunov_spectrum"], dtype=float).reshape(-1)
                if spec.size == 3:
                    lyap_spectra[vec_index, k, :] = spec
                else:
                    logger.warning(
                        f"[vec {vec_index}] unexpected spectrum size: {spec.size}"
                    )
                sim_status[vec_index, k] = res.get("sim_status", "Normal")
                intrinsic_dims[vec_index, k] = float(res["intrinsic_dimension_twonn"])
                kaplan_yorke_dims[vec_index, k] = float(res["kaplan_yorke_dimension"])
            except Exception as e:
                logger.exception(
                    f"[vec {vec_index}] features FAILED for init_{k}={init_point}: {e}"
                )
                per_init_results.append(None)
                sim_status[vec_index, k] = "FAIL"
        summary = summarize_over_inits(per_init_results)
        save_json(os.path.join(vec_dir, "features_summary.json"), summary)
        logger.info(f"[vec {vec_index}] features summary saved: {summary}")

    def _is_exploded(s):
        if s is None:
            return False
        s = str(s)
        return s.startswith("Exploded") or "Out of Bounds" in s or s == "FAIL"

    def _is_converged(s):
        if s is None:
            return False
        s = str(s)
        return s.startswith("Converged")

    nticks = min(6, num_vecs)
    tick_pos = np.linspace(0, num_vecs - 1, nticks).astype(int)
    tick_lbl = [f"{alphas[i] * 100:.0f}%" for i in tick_pos]
    for k, init_point in enumerate(INIT_POINTS):
        status_k = sim_status[:, k]
        explode_idx = np.where(np.vectorize(_is_exploded)(status_k))[0]
        conv_idx = np.where(np.vectorize(_is_converged)(status_k))[0]
        H = lyap_spectra[:, k, :].T
        M = np.ma.masked_invalid(H)
        id_line = intrinsic_dims[:, k]
        ky_line = kaplan_yorke_dims[:, k]
        (fig, axes) = plt.subplots(
            3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        ax = axes[0]
        im = ax.imshow(M, origin="lower", aspect="equal")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"])
        ax.set_title(f"Lyapunov spectrum heatmap (init_{k}={init_point})")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        if explode_idx.size > 0:
            ax.scatter(
                explode_idx,
                np.ones_like(explode_idx),
                marker="x",
                transform=ax.get_xaxis_transform(),
            )
        if conv_idx.size > 0:
            ax.scatter(
                conv_idx,
                np.ones_like(conv_idx),
                marker="o",
                facecolors="none",
                transform=ax.get_xaxis_transform(),
            )
        ax = axes[1]
        ax.plot(id_line)
        ax.set_ylabel("Intrinsic dim (TwoNN)")
        ax.grid(True, alpha=0.3)
        if explode_idx.size > 0:
            ax.scatter(
                explode_idx,
                np.ones_like(explode_idx),
                marker="x",
                transform=ax.get_xaxis_transform(),
            )
        if conv_idx.size > 0:
            ax.scatter(
                conv_idx,
                np.ones_like(conv_idx),
                marker="o",
                facecolors="none",
                transform=ax.get_xaxis_transform(),
            )
        ax = axes[2]
        ax.plot(ky_line)
        ax.set_ylabel("Kaplan-Yorke dim")
        ax.grid(True, alpha=0.3)
        if explode_idx.size > 0:
            ax.scatter(
                explode_idx,
                np.ones_like(explode_idx),
                marker="x",
                transform=ax.get_xaxis_transform(),
            )
        if conv_idx.size > 0:
            ax.scatter(
                conv_idx,
                np.ones_like(conv_idx),
                marker="o",
                facecolors="none",
                transform=ax.get_xaxis_transform(),
            )
        axes[2].set_xticks(tick_pos)
        axes[2].set_xticklabels(tick_lbl)
        axes[2].set_xlabel("weight (alpha)")
        plt.tight_layout()
        out_path = os.path.join(PIC_ROOT_PATH, f"init_{k}_features.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved per-init feature figure to {out_path}")


if __name__ == "__main__":
    main()
