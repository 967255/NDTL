import os
import json
from pathlib import Path

import numpy as np
import torch

from model import CAE
from ode_bank import *          # lorenz, chen, lv, ...
from utils_ode import *         # calc_grids, ode2vec, ...


# =========================
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "./checkpoints/cae_20260225_205314lorenz_chen_lv_ori.pt"
META_INFO = "lorenz_chen_lv_ori"                                     

ROOT_PATH = "./results_lirt"
SAVE_ROOT = os.path.join(ROOT_PATH, f"LIRT_pre_noise_{META_INFO}")
os.makedirs(SAVE_ROOT, exist_ok=True)

CAE_KWARGS = {
    # "cae_depth": 64,
    # "cae_input_dimension": 3,
    # "cae_strides": [3, 2, 2],
    # "cae_kernel_size": [5, 3, 3],
}

MODEL_CONFIG = {
    "grid_shape": [48, 48, 48],
    "grid_limits": [[-25, 25], [-25, 25], [0, 50]],
    "turb_scale": 0.0,
}

LIRT_CONFIG = {
    "protocol": "LIRT-pre-noise-v1",
    "feature_type": "pre_activation",
    "noise_type": "gaussian",
    # "eps_list": [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    "eps_list": [0.0,
            0.001, 0.002, 0.003, 0.005, 0.01,
            0.02, 0.03, 0.05, 0.1,
            0.2, 0.3, 0.5, 1.0],
    "num_repeats": 10,              # only used for epsilon > 0
    "use_common_random_numbers": True,
    "base_seed": 20260225,
    "save_compressed": True,
}

TEST_SYSTEM = lorenz


# =========================
# Utilities
# =========================
def eps_to_tag(eps: float) -> str:
    """0.003 -> eps_0p003 ; 1.0 -> eps_1"""
    s = f"{eps:.10g}"  # compact but stable
    s = s.replace(".", "p").replace("-", "m")
    return f"eps_{s}"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_torch_load(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def tensor_to_vec_numpy(vec_tensor_5d: torch.Tensor) -> np.ndarray:
    """
    [1, C, D, H, W] -> [D, H, W, C]
    """
    x = vec_tensor_5d.detach().cpu().squeeze(0).numpy()   # [C, D, H, W]
    x = np.transpose(x, (1, 2, 3, 0))                     # [D, H, W, C]
    return x


def compute_rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t.detach() ** 2)).item())


def build_system_tensor(system, cae_dtype=torch.float32):
    """
    Build one deterministic vector-field tensor for the given system.
    Returns:
        grids
        vec_np: numpy array [D,H,W,C] (as returned by ode2vec)
        vec_tensor: torch tensor [1,C,D,H,W]
    """
    grids = calc_grids(
        grid_limits=MODEL_CONFIG["grid_limits"],
        grid_shape=MODEL_CONFIG["grid_shape"]
    )
    vec_np = ode2vec(system, grids)                       # expected [D,H,W,C]
    vec_chw = np.transpose(vec_np, (3, 0, 1, 2))         # [C,D,H,W]
    vec_tensor = torch.tensor(vec_chw, dtype=cae_dtype).unsqueeze(0).to(DEVICE)
    return grids, vec_np, vec_tensor


@torch.no_grad()
def forward_clean_collect_preacts(cae: CAE, x: torch.Tensor):
    """
    Manual forward pass to collect encoder pre-activations and clean reconstruction.
    Returns:
        preacts: list of tensors (each [1,C,D,H,W], detached)
        z_final: latent tensor after last ReLU
        x_hat: clean reconstructed tensor [1,C,D,H,W]
    """
    z = x.clone()
    preacts = []

    # Encoder
    for layer in cae.encoder:
        h = layer(z)                  # pre-activation
        preacts.append(h.detach().clone())
        z = torch.relu(h)             # post-activation (same as model activation=ReLU)

    z_final = z.detach().clone()

    # Decoder
    x_hat = z
    for i_layer, layer in enumerate(cae.decoder):
        x_hat = layer(x_hat)
        if i_layer < len(cae.decoder) - 1:
            x_hat = torch.relu(x_hat)

    return preacts, z_final, x_hat


@torch.no_grad()
def forward_with_preact_noise(
    cae: CAE,
    x: torch.Tensor,
    target_layer: int,
    epsilon: float,
    layer_rms: float,
    noise_template: torch.Tensor = None,
):
    """
    LIRT-pre-noise:
      inject noise on encoder pre-activation h_l, then continue encoder+decoder.

    Args:
        x: [1,C,D,H,W]
        target_layer: encoder layer index
        epsilon: nominal noise strength
        layer_rms: clean pre-activation RMS (scalar, for normalization)
        noise_template: same shape as target pre-activation, standard Gaussian template.
                        If None and epsilon>0, sample on the fly.
    Returns:
        x_hat: [1,C,D,H,W]
        run_meta: dict with realized noise stats
    """
    z = x.clone()
    realized_noise_rms = 0.0
    realized_delta_h_over_h = None
    target_shape = None

    for i_layer, layer in enumerate(cae.encoder):
        h = layer(z)

        if i_layer == target_layer and epsilon > 0.0:
            target_shape = list(h.shape)

            if noise_template is None:
                noise_template = torch.randn_like(h)
            else:
                noise_template = noise_template.to(device=h.device, dtype=h.dtype)

            noise = epsilon * layer_rms * noise_template
            h_before = h
            h = h + noise

            realized_noise_rms = compute_rms(noise)
            denom = torch.sqrt(torch.mean(h_before ** 2)).item()
            numer = torch.sqrt(torch.mean((h - h_before) ** 2)).item()
            realized_delta_h_over_h = float(numer / (denom + 1e-12))

        z = torch.relu(h)

    # Decoder
    x_hat = z
    for i_layer, layer in enumerate(cae.decoder):
        x_hat = layer(x_hat)
        if i_layer < len(cae.decoder) - 1:
            x_hat = torch.relu(x_hat)

    run_meta = {
        "target_layer": int(target_layer),
        "epsilon_nominal": float(epsilon),
        "layer_rms": float(layer_rms),
        "realized_noise_rms": float(realized_noise_rms),
        "realized_delta_h_over_h": (
            None if realized_delta_h_over_h is None else float(realized_delta_h_over_h)
        ),
        "target_preact_shape": target_shape,
    }
    return x_hat, run_meta


def save_npz(file_path, **arrays):
    ensure_dir(Path(file_path).parent)
    if LIRT_CONFIG["save_compressed"]:
        np.savez_compressed(file_path, **arrays)
    else:
        np.savez(file_path, **arrays)


def save_json(file_path, obj):
    ensure_dir(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_noise_bank(clean_preacts, num_repeats, base_seed=0):
    """
    Common Random Numbers (CRN):
    For each layer and seed_id, generate one standard Gaussian template and reuse across epsilons.
    Store on CPU to save GPU memory.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(base_seed))

    noise_bank = []
    for i_layer, h in enumerate(clean_preacts):
        layer_bank = []
        shape = tuple(h.shape)
        for seed_id in range(num_repeats):
            noise = torch.randn(shape, generator=g, dtype=torch.float32, device="cpu")
            layer_bank.append(noise)
        noise_bank.append(layer_bank)
    return noise_bank


# =========================
# Main LIRT generator
# =========================
def lirt_pre_noise_generator(system):
    # --- system sanity ---
    if hasattr(system, "reset_coef"):
        try:
            system.reset_coef()
        except Exception:
            pass

    # --- build input ---
    cae_dtype = next(cae.parameters()).dtype
    grids, vec_np, x = build_system_tensor(system, cae_dtype=cae_dtype)

    # --- prepare dirs ---
    run_dir = os.path.join(SAVE_ROOT, system.name)
    baseline_dir = os.path.join(run_dir, "baseline")
    samples_dir = os.path.join(run_dir, "samples")
    ensure_dir(baseline_dir)
    ensure_dir(samples_dir)

    # --- clean forward, collect pre-activations ---
    preacts_clean, z_clean, x_hat_clean = forward_clean_collect_preacts(cae, x)
    n_layers = len(preacts_clean)
    layer_rms_list = [compute_rms(h) for h in preacts_clean]
    layer_shapes = [list(h.shape) for h in preacts_clean]

    # --- baseline stats ---
    clean_recon_mse = float(torch.mean((x_hat_clean - x) ** 2).item())

    # --- save input/original and baseline reconstruction ---
    save_npz(
        os.path.join(baseline_dir, "original_vec.npz"),
        vec=vec_np.astype(np.float32),
    )
    save_npz(
        os.path.join(baseline_dir, "clean_reconstruction.npz"),
        vec_hat=tensor_to_vec_numpy(x_hat_clean).astype(np.float32),
    )

    baseline_meta = {
        "system_name": system.name,
        "grid_shape": MODEL_CONFIG["grid_shape"],
        "grid_limits": MODEL_CONFIG["grid_limits"],
        "clean_reconstruction_mse": clean_recon_mse,
        "num_encoder_layers": int(n_layers),
        "layer_preact_rms": [float(v) for v in layer_rms_list],
        "layer_preact_shapes": layer_shapes,
    }
    save_json(os.path.join(baseline_dir, "baseline_meta.json"), baseline_meta)

    # --- noise bank (CRN) ---
    noise_bank = build_noise_bank(
        clean_preacts=preacts_clean,
        num_repeats=LIRT_CONFIG["num_repeats"],
        base_seed=LIRT_CONFIG["base_seed"],
    )

    # --- manifest top-level ---
    manifest = {
        "protocol": LIRT_CONFIG["protocol"],
        "feature_type": LIRT_CONFIG["feature_type"],
        "noise_type": LIRT_CONFIG["noise_type"],
        "use_common_random_numbers": bool(LIRT_CONFIG["use_common_random_numbers"]),
        "base_seed": int(LIRT_CONFIG["base_seed"]),
        "eps_list": [float(e) for e in LIRT_CONFIG["eps_list"]],
        "num_repeats_for_positive_eps": int(LIRT_CONFIG["num_repeats"]),
        "device": str(DEVICE),
        "checkpoint_path": CKPT_PATH,
        "meta_info": META_INFO,
        "model_config": MODEL_CONFIG,
        "cae_kwargs": CAE_KWARGS,
        "system_name": system.name,
        "baseline": {
            "original_vec_file": str(Path("baseline") / "original_vec.npz"),
            "clean_reconstruction_file": str(Path("baseline") / "clean_reconstruction.npz"),
            "baseline_meta_file": str(Path("baseline") / "baseline_meta.json"),
            "clean_reconstruction_mse": clean_recon_mse,
        },
        "layer_info": [
            {
                "layer_id": int(i),
                "preact_rms": float(layer_rms_list[i]),
                "preact_shape": layer_shapes[i],
            }
            for i in range(n_layers)
        ],
        "entries": [],
    }

    # --- save epsilon=0 baseline reference entries (one per layer for convenience) ---
    for layer_id in range(n_layers):
        manifest["entries"].append({
            "layer_id": int(layer_id),
            "epsilon_nominal": 0.0,
            "seed_id": None,
            "file": str(Path("baseline") / "clean_reconstruction.npz"),
            "kind": "baseline_reference",
        })

    # --- main loop: layer × epsilon × seed ---
    eps_list = [float(e) for e in LIRT_CONFIG["eps_list"]]
    num_repeats = int(LIRT_CONFIG["num_repeats"])

    for layer_id in range(n_layers):
        layer_rms = float(layer_rms_list[layer_id])

        for eps in eps_list:
            if eps == 0.0:
                continue

            eps_tag = eps_to_tag(eps)
            eps_dir = os.path.join(samples_dir, f"layer_{layer_id:02d}", eps_tag)
            ensure_dir(eps_dir)

            for seed_id in range(num_repeats):
                noise_template = noise_bank[layer_id][seed_id] if LIRT_CONFIG["use_common_random_numbers"] else None

                x_hat_noisy, run_meta = forward_with_preact_noise(
                    cae=cae,
                    x=x,
                    target_layer=layer_id,
                    epsilon=eps,
                    layer_rms=layer_rms,
                    noise_template=noise_template,
                )

                recon_mse = float(torch.mean((x_hat_noisy - x) ** 2).item())
                vec_hat_np = tensor_to_vec_numpy(x_hat_noisy).astype(np.float32)

                sample_filename = f"seed_{seed_id:02d}.npz"
                sample_relpath = Path("samples") / f"layer_{layer_id:02d}" / eps_tag / sample_filename
                sample_abspath = os.path.join(run_dir, str(sample_relpath))

                save_npz(
                    sample_abspath,
                    vec_hat=vec_hat_np,
                )

                entry = {
                    "kind": "noisy_reconstruction",
                    "layer_id": int(layer_id),
                    "epsilon_nominal": float(eps),
                    "seed_id": int(seed_id),
                    "file": str(sample_relpath),
                    "reconstruction_mse_vs_input": recon_mse,
                    "layer_rms": float(layer_rms),
                    "realized_noise_rms": float(run_meta["realized_noise_rms"]),
                    "realized_delta_h_over_h": run_meta["realized_delta_h_over_h"],
                    "target_preact_shape": run_meta["target_preact_shape"],
                }
                manifest["entries"].append(entry)

                print(
                    f"[LIRT] layer={layer_id:02d}, eps={eps:<6g}, seed={seed_id:02d} | "
                    f"recon_mse={recon_mse:.6e}, noise_rms={run_meta['realized_noise_rms']:.6e}"
                )

    # --- save manifest ---
    save_json(os.path.join(run_dir, "manifest.json"), manifest)
    print(f"\nDone. LIRT data saved to: {run_dir}")


# =========================
# Load CAE
# =========================
cae = CAE(**CAE_KWARGS).to(DEVICE)

ckpt = safe_torch_load(CKPT_PATH, DEVICE)
state_dict = extract_state_dict(ckpt)

print("Loading checkpoint...")
cae.load_state_dict(state_dict)
cae.eval()
print("Done!")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    lirt_pre_noise_generator(TEST_SYSTEM)