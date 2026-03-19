import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from ode_bank import *
from utils_ode import calc_grids, ode2vec, ensemble_rescaling
from utils_ode import time_reverse_ode, permute_sign_conjugacy, rotate_conjugacy


# -----------------------
# Config
# -----------------------
GRID_SHAPE  = [48, 48, 48]
GRID_LIMITS = [[0, 1], [0, 1], [0, 1]]
OUT_DIR = "./results_display/feat_results"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------
# Helpers: Gram + distances
# -----------------------
def gram_from_vec(vec_dhwc, normalize="none", eps=1e-6):
    """
    vec_dhwc: (D,H,W,C)
    normalize:
      - "none": raw Gram (scale-sensitive)
      - "zscore": channel-wise standardize over space (correlation-like)
    return: (C,C)
    """
    F = vec_dhwc.reshape(-1, vec_dhwc.shape[-1]).astype(np.float64)  # (N,C)
    if normalize == "zscore":
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True)
        sd = np.maximum(sd, eps)
        F = (F - mu) / sd
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'none' or 'zscore'")

    N = F.shape[0]
    G = (F.T @ F) / N
    return G


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def style_mse_from_grams(Ga, Gb, align_channels=False):
    """
    style distance between Gram matrices.
    If align_channels=True, minimize over channel permutations (good for variable permutation transforms).
    """
    C = Ga.shape[0]
    if not align_channels:
        return mse(Ga, Gb)

    best = float("inf")
    for p in itertools.permutations(range(C)):
        P = np.eye(C)[list(p)]
        d = mse(Ga, P @ Gb @ P.T)
        if d < best:
            best = d
    return best


def rotation_matrix_z(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0,0.0,1.0]], dtype=float)


# -----------------------
# Main experiment: per-system transform invariance
# -----------------------
def build_transforms(ode):
    """
    Return dict: transform_name -> transformed_ode
    """
    T = {}
    T["time_reverse"] = time_reverse_ode(ode)

    # variable permutation (swap x,y), no sign flip
    T["swap_xy"] = permute_sign_conjugacy(ode, perm=(1, 0, 2), signs=(1, 1, 1))

    # sign flip (reflect x)
    T["flip_x"]  = permute_sign_conjugacy(ode, perm=(0, 1, 2), signs=(-1, 1, 1))

    # rotation conjugacy around z (30 degrees)
    Rz = rotation_matrix_z(np.deg2rad(30.0))
    T["rot_z30"] = rotate_conjugacy(ode, R=Rz)

    return T


def run():
    # 1) choose base systems (rescaled)
    _, lorenz_u = ensemble_rescaling(lorenz, target_grid_limits=GRID_LIMITS, target_velocity=5.0)
    _, chen_u   = ensemble_rescaling(chen,   target_grid_limits=GRID_LIMITS, target_velocity=5.0)
    _, lv_u     = ensemble_rescaling(lv,     target_grid_limits=GRID_LIMITS, target_velocity=5.0)
    _, ross_u   = ensemble_rescaling(rossler,target_grid_limits=GRID_LIMITS, target_velocity=5.0)

    base_odes = [lorenz_u, chen_u, lv_u, ross_u]

    grids = calc_grids(grid_limits=GRID_LIMITS, grid_shape=GRID_SHAPE)

    rows = []
    xs = []  # for scatter: content
    ys = []  # for scatter: style

    for ode in base_odes:
        # sample base vec
        V = ode2vec(ode, grids)  # (D,H,W,3)

        # precompute base grams
        G0_raw = gram_from_vec(V, normalize="none")
        G0_z   = gram_from_vec(V, normalize="zscore")

        transforms = build_transforms(ode)

        # energy scale for a relative content metric (optional but helpful)
        energy = float(np.mean(V**2)) + 1e-12

        for tname, ode_t in transforms.items():
            Vt = ode2vec(ode_t, grids)

            # content (absolute & relative)
            c_abs = mse(V, Vt)
            c_rel = c_abs / energy

            # grams
            Gt_raw = gram_from_vec(Vt, normalize="none")
            Gt_z   = gram_from_vec(Vt, normalize="zscore")

            # for permutation-like transforms, align_channels=True helps
            align = (tname in ["swap_xy"])

            s_raw = style_mse_from_grams(G0_raw, Gt_raw, align_channels=align)
            s_z   = style_mse_from_grams(G0_z,   Gt_z,   align_channels=align)

            rows.append([ode.name, tname, c_abs, c_rel, s_raw, s_z])
            xs.append(c_rel)
            ys.append(s_z)

            print(f"{ode.name:20s} | {tname:10s} | content_rel={c_rel:.3e} | style_z={s_z:.3e}")

    rows = np.array(rows, dtype=object)

    # 2) save CSV-like text
    out_txt = os.path.join(OUT_DIR, "transform_content_style.txt")
    with open(out_txt, "w") as f:
        f.write("system\ttransform\tcontent_abs\tcontent_rel\tstyle_raw\tstyle_z\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print("Saved:", out_txt)

    # 3) scatter plot: content vs style
    # ---- replace the scatter block with this ----
    def heatmap_with_text(A, row_names, col_names, title, save_path, fmt="{:.2g}"):
        plt.figure(figsize=(7.5, 2.8), dpi=220)
        plt.imshow(A, interpolation="nearest", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(col_names)), col_names, rotation=45, ha="right")
        plt.yticks(range(len(row_names)), row_names)
        plt.title(title)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                plt.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()


    # rows is a python list of [system, transform, c_abs, c_rel, s_raw, s_z]
    systems = sorted({r[0] for r in rows})
    transforms = ["time_reverse", "swap_xy", "flip_x", "rot_z30"]

    content_rel_mat = np.full((len(systems), len(transforms)), np.nan, dtype=float)
    style_z_mat     = np.full((len(systems), len(transforms)), np.nan, dtype=float)

    for sys, tname, c_abs, c_rel, s_raw, s_z in rows:
        i = systems.index(sys)
        j = transforms.index(tname)
        content_rel_mat[i, j] = float(c_rel)
        style_z_mat[i, j]     = float(s_z)

    score_mat = style_z_mat / (content_rel_mat + 1e-12)

    heatmap_with_text(content_rel_mat, systems, transforms,
                    "Content change (content_rel)", os.path.join(OUT_DIR, "content_rel_heatmap.png"),
                    fmt="{:.2g}")
    heatmap_with_text(style_z_mat, systems, transforms,
                    "Style change (style_z)", os.path.join(OUT_DIR, "style_z_heatmap.png"),
                    fmt="{:.2g}")
    heatmap_with_text(score_mat, systems, transforms,
                    "Invariance score = style_z / content_rel", os.path.join(OUT_DIR, "invariance_score.png"),
                    fmt="{:.2e}")
    # ---- end replace ----



if __name__ == "__main__":
    run()
