import json, os, logging

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CAE
from ode_bank import *
from utils_eval import FeatureConfig, compute_system_features, summarize_over_inits, save_json
from utils_ode import vec2ode


'''CONFIG'''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# !!!!!!!change
VEC_PATH = "./results/fusion_20260303_190210lorenz_chen_lv_ori/vecs_lorenz_lv_fusion_20260303_190210lorenz_chen_lv_ori.npy"
CAE_INFO = 'lorenz_chen_lv_ori'
META_INFO = 'lorenz_c+lv_c'
ROOT_PATH = './results_display/fusion_results'
PIC_ROOT_PATH = ROOT_PATH + '/' + CAE_INFO + '/' + META_INFO
os.makedirs(PIC_ROOT_PATH, exist_ok=True)


'''MODEL CONFIG'''
grid_shape = [48, 48, 48] 
grid_limits = [[-25,25], [-25,25], [0,50]] 

# INIT_POINTS = [
#     (0.3, 0.3, 0.5),
#     (0.1, 0.2, 0.8),
#     (0.1, 0.0, 0.0),
# ]
# !!!!!!!change
INIT_POINTS = [
    (15, 15, 25),
    (5, 10, 40),
    (5, 0.0, 0.0),
]


'''LOGGER CONFIG'''
log_file = os.path.join(PIC_ROOT_PATH, 'info.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# NEW: feature config (shared across all inits/vecs; you can tune here)
FEATURE_CFG = FeatureConfig(
    traj_length=5000,
    omit_points=0,
    h=None,                 # None -> use ode_hat.h
    burn_in=500,
    stride=5,
    twonn_approximate=False,
    twonn_theiler=10,
    twonn_k_search=64,
    lyap_tol=1e-8,
    lyap_min_tpts=10,
    lyap_fd_eps=None,
    save_traj_npz=True,
    save_lyap_npz=True,
    save_plots=True,
)


'''BODY'''
vecs = np.load(VEC_PATH)
logger.info(f"vecs.shape = {vecs.shape}")   # (N, 1, 48, 48, 48, 3)

grids = calc_grids(grid_limits=grid_limits, grid_shape=grid_shape)

num_vecs = vecs.shape[0]
logger.info(f"num_vecs = {num_vecs}")


# ========== NEW: collect spectra/status for visualization ==========
K = len(INIT_POINTS)      # number of initial conditions
D = 3                     # ode dimension (x,y,z) -> 3 exponents

# lyap_spectra, id dim and ky dim [vec_index, init_k, d]
lyap_spectra = np.full((num_vecs, K, D), np.nan, dtype=float)
sim_status = np.empty((num_vecs, K), dtype=object)
intrinsic_dims = np.full((num_vecs, K), np.nan, dtype=float)   # ID per (vec, init)
kaplan_yorke_dims = np.full((num_vecs, K), np.nan, dtype=float) # KY per (vec, init)


# weight axis: chen percentage alpha in [0,1]
# 如果你的 vec0=0% chen, vec_last=100% chen
alphas = np.linspace(0.0, 1.0, num_vecs)


for vec_index in range(num_vecs):
    logger.info(f"=== start vec index {vec_index} ===")

    vec = vecs[vec_index]   # shape: (1, 48, 48, 48, 3)
    logger.info(f"vec[{vec_index}].shape = {vec.shape}")

    vec_dir = os.path.join(PIC_ROOT_PATH, f"vec_{vec_index}")
    os.makedirs(vec_dir, exist_ok=True)

    ode_hat = vec2ode(
        vec, 
        grids, 
        sparse_coef=None, 
        max_order=2, 
        extra_term_dicts=None,
        mode='linear', 
        alpha=0.1, 
        alphas=None, 
        cv=5, 
        fit_intercept=True,
        name=f'ode_hat_{vec_index}',
        verbose=False, 
        return_diagnostics=False, 
        grid_limits=None
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
    config_path = os.path.join(vec_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"[vec {vec_index}] config saved to {config_path}")

    info = ode_hat.get_info()
    logger.info(f"[vec {vec_index}] ode_hat info: {info}")

    info_path = os.path.join(vec_dir, 'ode_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(str(info))
    logger.info(f"[vec {vec_index}] ode info saved to {info_path}")

    # ---- per-init features ----
    per_init_results = []
    for k, init_point in enumerate(INIT_POINTS):
        init_dir = os.path.join(vec_dir, f"init_{k}")
        os.makedirs(init_dir, exist_ok=True)

        # system display
        ode_hat.plot(pic_save_dir=os.path.join(init_dir, "trajectory"), init_point=init_point)
        logger.info(f"[vec {vec_index}] trajectory for init_{k}={init_point} saved under {init_dir}")

        # system features computaion
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
                f"[vec {vec_index}] features done for init_{k}={init_point}: "
                f"ID={res['intrinsic_dimension_twonn']:.4f}, "
                f"KY={res['kaplan_yorke_dimension']:.4f}, "
                f"spectrum={np.array(res['lyapunov_spectrum'])}"
            )
            # ========== store for global heatmap ==========
            spec = np.asarray(res["lyapunov_spectrum"], dtype=float).reshape(-1)
            if spec.size == 3:
                lyap_spectra[vec_index, k, :] = spec
            else:
                logger.warning(f"[vec {vec_index}] unexpected spectrum size: {spec.size}")
            sim_status[vec_index, k] = res.get("sim_status", "Normal")
            intrinsic_dims[vec_index, k] = float(res["intrinsic_dimension_twonn"])
            kaplan_yorke_dims[vec_index, k] = float(res["kaplan_yorke_dimension"])


        except Exception as e:
            logger.exception(f"[vec {vec_index}] features FAILED for init_{k}={init_point}: {e}")
            per_init_results.append(None)
            sim_status[vec_index, k] = "FAIL"


    # ---- NEW: vec-level summary over inits ----
    summary = summarize_over_inits(per_init_results)
    save_json(os.path.join(vec_dir, "features_summary.json"), summary)
    logger.info(f"[vec {vec_index}] features summary saved: {summary}")


def _is_exploded(s):
    if s is None:
        return False
    s = str(s)
    return s.startswith("Exploded") or ("Out of Bounds" in s) or (s == "FAIL")

def _is_converged(s):
    if s is None:
        return False
    s = str(s)
    return s.startswith("Converged")

# x-axis ticks labeled by alpha (chen weight)
nticks = min(6, num_vecs)
tick_pos = np.linspace(0, num_vecs - 1, nticks).astype(int)
tick_lbl = [f"{alphas[i]*100:.0f}%" for i in tick_pos]

for k, init_point in enumerate(INIT_POINTS):
    status_k = sim_status[:, k]
    explode_idx = np.where(np.vectorize(_is_exploded)(status_k))[0]
    conv_idx = np.where(np.vectorize(_is_converged)(status_k))[0]

    # ---- spectrum heatmap for this init ----
    H = lyap_spectra[:, k, :].T   # (3, num_vecs)
    M = np.ma.masked_invalid(H)

    # ---- line data for this init ----
    id_line = intrinsic_dims[:, k]       # (num_vecs,)
    ky_line = kaplan_yorke_dims[:, k]    # (num_vecs,)

    fig, axes = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # (a) spectrum heatmap
    ax = axes[0]
    # im = ax.imshow(M, aspect="auto", origin="lower")
    im = ax.imshow(M, origin="lower", aspect="equal")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
    ax.set_title(f"Lyapunov spectrum heatmap (init_{k}={init_point})")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    if explode_idx.size > 0:
        ax.scatter(explode_idx, np.ones_like(explode_idx), marker="x",
                   transform=ax.get_xaxis_transform())
    if conv_idx.size > 0:
        ax.scatter(conv_idx, np.ones_like(conv_idx), marker="o", facecolors="none",
                   transform=ax.get_xaxis_transform())

    # (b) intrinsic dimension line
    ax = axes[1]
    ax.plot(id_line)
    ax.set_ylabel("Intrinsic dim (TwoNN)")
    ax.grid(True, alpha=0.3)

    if explode_idx.size > 0:
        ax.scatter(explode_idx, np.ones_like(explode_idx), marker="x",
                   transform=ax.get_xaxis_transform())
    if conv_idx.size > 0:
        ax.scatter(conv_idx, np.ones_like(conv_idx), marker="o", facecolors="none",
                   transform=ax.get_xaxis_transform())

    # (c) Kaplan–Yorke dimension line
    ax = axes[2]
    ax.plot(ky_line)
    ax.set_ylabel("Kaplan-Yorke dim")
    ax.grid(True, alpha=0.3)

    if explode_idx.size > 0:
        ax.scatter(explode_idx, np.ones_like(explode_idx), marker="x",
                   transform=ax.get_xaxis_transform())
    if conv_idx.size > 0:
        ax.scatter(conv_idx, np.ones_like(conv_idx), marker="o", facecolors="none",
                   transform=ax.get_xaxis_transform())

    axes[2].set_xticks(tick_pos)
    axes[2].set_xticklabels(tick_lbl)
    axes[2].set_xlabel("weight (alpha)")

    plt.tight_layout()
    out_path = os.path.join(PIC_ROOT_PATH, f"init_{k}_features.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-init feature figure to {out_path}")
