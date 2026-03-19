from __future__ import annotations

import torch, numpy as np, matplotlib.pyplot as plt
import warnings
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



# =============================
# display latent feature
# =============================
def gram_matrix(x):
    if len(x.shape) == 4:
        c, d, h, w = x.size()
        b = 1
    elif len(x.shape) == 5:
        b, c, d, h, w = x.size()
    else:
        raise ValueError(f'Input tensor shape {x.shape} not supported!')
    
    target_features = x.view(b, c, d * h * w)  # flatten the feature maps
    gram_product = torch.bmm(target_features, target_features.transpose(1, 2))  # compute the gram product
    return gram_product.div(c * d * h * w)

def show_tensor_slices(
    x: torch.Tensor,
    titles=None,
    norm: str = "global",     # "global" | "row" | "local"
    cbar: str = "row",        # "figure" | "row" | "each"
    vmin: float = None,
    vmax: float = None,
    slice_idx: list = None,
    save_path: str = None
):
    if x.dim() == 4:   # (3,D,H,W) -> (1,3,D,H,W)
        x = x.unsqueeze(0)
    elif x.dim() != 5:
        raise ValueError(f"Expected (B,3,D,H,W) or (3,D,H,W), got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    # (B,3,D,H,W) -> (B,D,H,W,3)
    V = np.moveaxis(x.detach().cpu().numpy(), 1, -1)  # (B,D,H,W,3)
    mag = np.linalg.norm(V, axis=-1)                  # (B,D,H,W)

    slices_all = []
    if slice_idx:
        for b in range(B):
            slices_all.append([
                mag[b, :, :, slice_idx[2]],  # axial (z=mid)
                mag[b, :, slice_idx[1], :],  # coronal (y=mid)
                mag[b, slice_idx[0], :, :],  # sagittal (x=mid)
            ])
    else:      
        for b in range(B):
            slices_all.append([
                mag[b, :, :, W//2],  # axial (z=mid)
                mag[b, :, H//2, :],  # coronal (y=mid)
                mag[b, D//2, :, :],  # sagittal (x=mid)
            ])

    if cbar == "figure" and norm != "global":
        warnings.warn('cbar="figure" need norm="global", have set norm=="global" auto. ')
        norm = "global"
    if cbar == "row" and norm == "local":
        warnings.warn('cbar="row" not match norm="local", cannot use a single colorbar, have set norm=="row"。')
        norm = "row"

    if norm == "global":
        if vmin is None:
            vmin_g = float(np.nanmin(mag))
        else:
            vmin_g = float(vmin)
        if vmax is None:
            vmax_g = float(np.nanmax(mag))
        else:
            vmax_g = float(vmax)

    elif norm == "row":
        row_min = []
        row_max = []
        for b in range(B):
            vals = np.concatenate([s.ravel() for s in slices_all[b]])
            row_min.append(float(np.nanmin(vals)))
            row_max.append(float(np.nanmax(vals)))

    else:  # "local"
        pass 

    fig, axes = plt.subplots(B, 3, figsize=(6, 2*max(1, B)))
    if B == 1:
        axes = np.expand_dims(axes, 0)  # -> (1,3)

    names = ["axial (z=mid)", "coronal (y=mid)", "sagittal (x=mid)"]
    ims = [] 

    for b in range(B):
        # row_title = titles[b] if (titles is not None and b < len(titles)) else f"sample{b}"
        for j in range(3):
            ax = axes[b, j]
            img = slices_all[b][j]

            if norm == "global":
                vmin_use, vmax_use = vmin_g, vmax_g
            elif norm == "row":
                vmin_use, vmax_use = row_min[b], row_max[b]
            else:  # "local"
                vmin_use, vmax_use = float(np.nanmin(img)), float(np.nanmax(img))

            im = ax.imshow(img, vmin=vmin_use, vmax=vmax_use)
            ims.append(im)
            # ax.set_title(f"{row_title} | {names[j]}")
            ax.axis("off")

            if cbar == "each":
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if cbar == "row":
            fig.colorbar(
                ims[-1], 
                ax=[axes[b, 0], axes[b, 1], axes[b, 2]],
                fraction=0.046, pad=0.04
            )


    if cbar == "figure":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=vmin_g, vmax=vmax_g), cmap=ims[-1].get_cmap())
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
        fig.colorbar(sm, cax=cbar_ax)

    # fig.tight_layout(rect=(0, 0, 0.9, 1) if cbar == "figure" else None)
    if save_path:
        plt.savefig(save_path, dpi=450)
    plt.show()
    return fig, axes

def show_gram(
    G,
    titles=None,
    norm: str = "global", 
    cbar: str = "figure", 
    vmin: float = None,
    vmax: float = None,
    show_ticks: bool = False, 
    save_path: str = None
):
    if isinstance(G, torch.Tensor):
        G = G.detach().cpu().numpy()
    assert G.ndim == 3, f"Expected (B,C,C), got {G.shape}"
    B, C, C2 = G.shape
    assert C == C2, "Last two dims must be equal (C,C)."

    if norm == "global":
        vmin_use = float(np.nanmin(G)) if vmin is None else float(vmin)
        vmax_use = float(np.nanmax(G)) if vmax is None else float(vmax)
        vmins = [vmin_use] * B
        vmaxs = [vmax_use] * B
    elif norm == "local":
        vmins = [float(np.nanmin(G[b])) for b in range(B)]
        vmaxs = [float(np.nanmax(G[b])) for b in range(B)]
    else:
        raise ValueError('norm must be "global" or "local"')

    fig, axes = plt.subplots(1, B, figsize=(4*B, 4), squeeze=False)
    axes = axes[0]  # 1×B

    ims = []
    for b in range(B):
        ax = axes[b]
        im = ax.imshow(G[b], vmin=vmins[b], vmax=vmaxs[b], interpolation="nearest")
        ims.append(im)
        if titles and b < len(titles):
            ax.set_title(titles[b])
        if show_ticks:
            ax.set_xlabel("channel"); ax.set_ylabel("channel")
        else:
            ax.set_xticks([]); ax.set_yticks([])

        if cbar == "each":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if cbar == "figure":
        if norm == "global":
            ref_im = ims[-1]
            cb = fig.colorbar(ref_im, ax=axes, fraction=0.046, pad=0.04)
            cb.mappable.set_clim(vmins[0], vmaxs[0])
        else:
            ref_im = ims[-1]
            vmin_g = float(np.nanmin(G)) if vmin is None else float(vmin)
            vmax_g = float(np.nanmax(G)) if vmax is None else float(vmax)
            ref_im.set_clim(vmin_g, vmax_g)
            for im in ims: im.set_clim(vmin_g, vmax_g)
            fig.colorbar(ref_im, ax=axes, fraction=0.046, pad=0.04)

    # fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=450)
    plt.show()
    return fig, axes



# =============================
# mathematics properties
# =============================
# =============================
# IO
# =============================
def _to_jsonable(x: Any):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x

def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_jsonable)

# =============================
# Jacobian via finite differences for ode.d
# =============================
def jac_fd_from_ode_d(ode: Any, x: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Central finite difference Jacobian for f(x)=ode.d(x).
    Your ode.d expects points shape (N,d) and returns (N,d).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.size

    if eps is None:
        eps = 1e-6 * (1.0 + np.linalg.norm(x))

    def f(xx: np.ndarray) -> np.ndarray:
        return np.asarray(ode.d(np.asarray(xx, dtype=float).reshape(1, -1)), dtype=float).reshape(-1)

    fx = f(x)
    if fx.size != d:
        raise ValueError(f"ode.d returned shape {fx.shape}, expected ({d},)")

    J = np.zeros((d, d), dtype=float)
    for j in range(d):
        dx = np.zeros(d, dtype=float)
        dx[j] = eps
        fp = f(x + dx)
        fm = f(x - dx)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J

# =============================
# TwoNN intrinsic dimension
# =============================
def twonn_dim(X: np.ndarray, approximate: bool = False) -> float:
    """
    Standard TwoNN global intrinsic dimension estimator (Facco et al., 2017).
    """
    X = np.asarray(X, dtype=float)

    if approximate:
        try:
            import hnswlib  # type: ignore
        except ImportError as e:
            raise ImportError("hnswlib not installed; set approximate=False or install hnswlib.") from e

        index = hnswlib.Index(space="l2", dim=X.shape[1])
        index.init_index(max_elements=X.shape[0], ef_construction=100, M=16)
        index.add_items(X)
        _, distances = index.knn_query(X, k=3)
        distances = np.sqrt(distances)
    else:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as e:
            raise ImportError("scikit-learn not installed; please install scikit-learn.") from e

        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = r2 / r1
    mu = mu[np.isfinite(mu) & (mu > 1)]
    if mu.size == 0:
        raise ValueError("TwoNN failed: no valid mu values.")
    return float(mu.size / np.sum(np.log(mu)))

def twonn_dim_timeseries(
    X: np.ndarray,
    theiler: int = 0,
    k_search: int = 64,
) -> float:
    """
    TwoNN for time-series point clouds with Theiler window:
    exclude neighbors within +/- theiler indices.

    NOTE: still TwoNN, just changes how r1/r2 are selected to avoid temporal neighbors.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    if N < 20:
        raise ValueError("Too few points for a stable TwoNN estimate.")

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("scikit-learn not installed; please install scikit-learn.") from e

    k = min(max(3, k_search), N)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    r1 = np.full(N, np.nan, dtype=float)
    r2 = np.full(N, np.nan, dtype=float)

    for i in range(N):
        cnt = 0
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):  # skip self
            if theiler > 0 and abs(int(j) - i) <= theiler:
                continue
            cnt += 1
            if cnt == 1:
                r1[i] = dist
            elif cnt == 2:
                r2[i] = dist
                break

    mu = r2 / r1
    mu = mu[np.isfinite(mu) & (mu > 1)]
    if mu.size == 0:
        raise ValueError("TwoNN(timeseries) failed: no valid neighbor pairs after Theiler exclusion.")
    return float(mu.size / np.sum(np.log(mu)))

# =============================
# Lyapunov spectrum (Christiansen & Rugh style; QR each step)
# =============================
@dataclass
class LyapunovResult:
    spectrum: np.ndarray          # (d,) sorted desc
    dt: float
    used_steps: int
    local_log_rdiag: np.ndarray   # (K, d) where K ~ used_steps-1

def find_lyapunov_exponents_for_ode(
    ode: Any,
    init_point: np.ndarray,
    traj_length: int,
    h: float | None = None,
    tol: float = 1e-8,
    min_tpts: int = 10,
    fd_eps: float | None = None,
    omit_points: int = 0,
) -> LyapunovResult:
    """
    Adapted version of your reference `find_lyapunov_exponents`,
    but directly using your ODE class:

      - trajectory from ode.exp(...)
      - rhs from ode.d(...)
      - Jacobian from FD (jac_fd_from_ode_d)

    Tangent update:
      U <- (I - J dt)^(-1) U   (backward Euler on tangent)
      QR -> accumulate log|diag(R)|
    """
    if h is None:
        h = float(getattr(ode, "h", 0.01))

    init_point = np.asarray(init_point, dtype=float).reshape(-1)
    d = int(getattr(ode, "dimension", init_point.size))

    # traj = ode.exp(init_point=init_point, num_points=traj_length, omit_points=omit_points, h=h)
    # traj = np.asarray(traj, dtype=float)
    traj, status = ode.exp(
        init_point=init_point,
        num_points=traj_length,
        omit_points=omit_points,
        h=h,
        return_status=True,
        epsilon=0.0,
    )
    traj = np.asarray(traj, dtype=float)

    if traj.shape[0] < 2:
        raise RuntimeError(f"Lyapunov: trajectory too short with status={status}")

    used_steps = int(traj.shape[0])
    if used_steps < 2:
        raise ValueError("Trajectory too short for Lyapunov computation.")

    dt = float(h)

    U = np.eye(d, dtype=float)
    logs: List[np.ndarray] = []

    for i in range(used_steps):
        if i < 1:
            continue

        x = traj[i]
        J = jac_fd_from_ode_d(ode, x, eps=fd_eps)

        A = np.eye(d) - J * dt
        try:
            U_new = np.linalg.solve(A, U)
        except np.linalg.LinAlgError:
            used_steps = i
            break

        Q, R = np.linalg.qr(U_new)
        logdiag = np.log(np.abs(np.diag(R)) + 1e-300)
        logs.append(logdiag)
        U = Q

        # optional early stop heuristic
        if (np.min(np.abs(logdiag)) < tol) and (i > min_tpts):
            used_steps = i + 1
            break

    if len(logs) == 0:
        raise RuntimeError("Lyapunov computation failed: no QR steps collected.")

    logs_arr = np.asarray(logs, dtype=float)  # (K, d)
    spectrum = np.sum(logs_arr, axis=0) / (dt * used_steps)
    spectrum = np.sort(spectrum)[::-1]
    return LyapunovResult(spectrum=spectrum, dt=dt, used_steps=used_steps, local_log_rdiag=logs_arr)

# =============================
# Kaplan–Yorke dimension
# =============================
def kaplan_yorke_dimension(spectrum0: np.ndarray) -> float:
    spectrum = np.sort(np.asarray(spectrum0, dtype=float))[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    idx = np.where(cspec >= 0)[0]
    if len(idx) == 0:
        return 0.0
    j = int(np.max(idx))
    if j > d - 2:
        j = d - 2
        warnings.warn(
            "Cumulative sum of Lyapunov exponents never crosses zero. "
            "System may be ill-posed or undersampled."
        )
    return float(1 + j + cspec[j] / (np.abs(spectrum[j + 1]) + 1e-300))

# =============================
# Plot hooks
# =============================
def plot_lyapunov_convergence(local_log_rdiag: np.ndarray, dt: float, save_path: str | Path) -> None:
    """
    Running average convergence of Lyapunov estimates.
    """
    import matplotlib.pyplot as plt

    local = np.asarray(local_log_rdiag, dtype=float)
    K, d = local.shape
    running = np.cumsum(local, axis=0) / (dt * np.arange(1, K + 1)[:, None])

    plt.figure()
    for j in range(d):
        plt.plot(running[:, j])
    plt.xlabel("step")
    plt.ylabel("running Lyapunov estimate")
    plt.title("Lyapunov convergence (running average)")
    plt.grid(True, alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_lyapunov_spectrum(spectrum: np.ndarray, save_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    s = np.asarray(spectrum, dtype=float).reshape(-1)
    xs = np.arange(len(s))
    plt.figure()
    plt.bar(xs, s)
    plt.xlabel("index (sorted desc)")
    plt.ylabel("Lyapunov exponent")
    plt.title("Lyapunov spectrum")
    plt.grid(True, axis="y", alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# =============================
# Feature config + runner
# =============================
@dataclass
class FeatureConfig:
    # trajectory length for both TwoNN and Lyapunov
    traj_length: int = 5000
    omit_points: int = 0
    h: float | None = None

    # TwoNN sampling from trajectory
    burn_in: int = 500
    stride: int = 5
    twonn_approximate: bool = False
    twonn_theiler: int = 0
    twonn_k_search: int = 64

    # Lyapunov
    lyap_tol: float = 1e-8
    lyap_min_tpts: int = 10
    lyap_fd_eps: float | None = None

    # Save switches
    save_traj_npz: bool = True
    save_lyap_npz: bool = True
    save_plots: bool = True

def compute_system_features(
    ode: Any,
    init_point: Tuple[float, ...] | List[float] | np.ndarray,
    out_dir: str | Path,
    cfg: FeatureConfig,
) -> Dict[str, Any]:
    """
    Compute and save:
      - TwoNN intrinsic dimension (on trajectory point cloud)
      - Lyapunov spectrum (QR)
      - Kaplan–Yorke dimension

    Files under out_dir:
      - feature_config.json
      - results.json
      - traj.npz (optional)
      - lyap_history.npz (optional)
      - figs/*.png (optional)
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "feature_config.json", asdict(cfg))

    init_point = np.asarray(init_point, dtype=float).reshape(-1)
    h = float(cfg.h) if cfg.h is not None else float(getattr(ode, "h", 0.01))

    # trajectory (once) for TwoNN + (optionally) saving
    # traj = ode.exp(init_point=init_point, num_points=cfg.traj_length, omit_points=cfg.omit_points, h=h)
    # traj = np.asarray(traj, dtype=float)
    # tpts = np.arange(traj.shape[0], dtype=float) * h
    traj, status = ode.exp(
        init_point=init_point,
        num_points=cfg.traj_length,
        omit_points=cfg.omit_points,
        h=h,
        return_status=True, 
        epsilon=0.0,        
    )
    traj = np.asarray(traj, dtype=float)
    tpts = np.arange(traj.shape[0], dtype=float) * h
    results_sim = {
        "sim_status": status,
        "sim_length": int(traj.shape[0]),
        "h": float(h),
    }

    MIN_LEN_FOR_FEATURES = 1000 

    if traj.shape[0] < MIN_LEN_FOR_FEATURES and status != "Normal":
        if isinstance(status, str) and status.startswith("Converged"):
            x_star = traj[-1]
            J = jac_fd_from_ode_d(ode, x_star, eps=cfg.lyap_fd_eps)
            eig = np.linalg.eigvals(J)
            spectrum = np.sort(np.real(eig))[::-1]

            results = {
                "init_point": init_point,
                **results_sim,
                "intrinsic_dimension_twonn": 0.0,
                "lyapunov_spectrum": spectrum,
                "kaplan_yorke_dimension": 0.0,
                "note": "Converged early; used fixed-point linearization for Lyapunov spectrum; set ID=0, KY=0.",
            }
            save_json(out_dir / "results.json", results)

            if cfg.save_traj_npz:
                np.savez_compressed(out_dir / "traj.npz", tpts=tpts, traj=traj, h=h)

            if cfg.save_plots:
                plot_lyapunov_spectrum(spectrum, fig_dir / "lyapunov_spectrum.png")

            return results

        raise RuntimeError(f"Trajectory ended early (len={traj.shape[0]}) with status={status}")


    # for TwoNN: burn-in & stride
    traj_used = traj[cfg.burn_in :: cfg.stride]
    if traj_used.shape[0] < 20:
        raise ValueError("traj_used too short; reduce burn_in/stride or increase traj_length.")

    if cfg.twonn_theiler > 0:
        intrinsic_dim = twonn_dim_timeseries(traj_used, theiler=cfg.twonn_theiler, k_search=cfg.twonn_k_search)
    else:
        intrinsic_dim = twonn_dim(traj_used, approximate=cfg.twonn_approximate)

    # Lyapunov spectrum
    lyap = find_lyapunov_exponents_for_ode(
        ode=ode,
        init_point=init_point,
        traj_length=cfg.traj_length,
        h=h,
        tol=cfg.lyap_tol,
        min_tpts=cfg.lyap_min_tpts,
        fd_eps=cfg.lyap_fd_eps,
        omit_points=cfg.omit_points,
    )
    dky = kaplan_yorke_dimension(lyap.spectrum)

    results = {
        "init_point": init_point,
        **results_sim,
        "intrinsic_dimension_twonn": float(intrinsic_dim),
        "lyapunov_spectrum": lyap.spectrum,
        "kaplan_yorke_dimension": float(dky),
        "dt": float(lyap.dt),
        "lyap_used_steps": int(lyap.used_steps),
        "traj_full_shape": traj.shape,
        "traj_used_shape": traj_used.shape,
    }
    save_json(out_dir / "results.json", results)

    if cfg.save_traj_npz:
        np.savez_compressed(out_dir / "traj.npz", tpts=tpts, traj=traj, traj_used=traj_used, h=h)

    if cfg.save_lyap_npz:
        np.savez_compressed(
            out_dir / "lyap_history.npz",
            local_log_rdiag=lyap.local_log_rdiag,
            spectrum=lyap.spectrum,
            dt=lyap.dt,
            used_steps=lyap.used_steps,
        )

    if cfg.save_plots:
        plot_lyapunov_convergence(lyap.local_log_rdiag, lyap.dt, fig_dir / "lyapunov_convergence.png")
        plot_lyapunov_spectrum(lyap.spectrum, fig_dir / "lyapunov_spectrum.png")

    return results

def summarize_over_inits(per_init_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple aggregation over multiple initial conditions:
    mean/std for intrinsic_dim, KY dim, and spectrum.
    """
    ok = [r for r in per_init_results if r is not None and "lyapunov_spectrum" in r]
    if len(ok) == 0:
        return {"n_success": 0}

    ids = np.array([r["intrinsic_dimension_twonn"] for r in ok], dtype=float)
    kys = np.array([r["kaplan_yorke_dimension"] for r in ok], dtype=float)
    specs = np.array([r["lyapunov_spectrum"] for r in ok], dtype=float)  # (n, d)

    return {
        "n_success": len(ok),
        "intrinsic_dimension_twonn_mean": float(np.mean(ids)),
        "intrinsic_dimension_twonn_std": float(np.std(ids)),
        "kaplan_yorke_dimension_mean": float(np.mean(kys)),
        "kaplan_yorke_dimension_std": float(np.std(kys)),
        "lyapunov_spectrum_mean": np.mean(specs, axis=0),
        "lyapunov_spectrum_std": np.std(specs, axis=0),
    }
