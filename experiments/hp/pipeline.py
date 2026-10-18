from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ndtl.models import EcologyCAE
from .dynamics import (
    TrilinearVectorField,
    integrate_rhs,
    lyapunov_spectrum,
    summarize_params,
    summarize_trajectory,
)
from .fit import fit_ecological_basis_linear, fit_holling_nonlinear
from .holling import (
    DEFAULT_DOMAIN,
    DOMAIN_PRESETS,
    HP_FALLBACK_MODULES,
    HollingParams,
    holling_rhs_ecology,
    parent_augmented_samples,
    tensor_to_torch_channels,
    torch_channels_to_tensor,
    vector_field_tensor,
)


@dataclass
class Profile:
    name: str
    grid_size: int
    depth: int
    strides: tuple[int, int, int]
    kernels: tuple[int, int, int]
    batch_size: int
    train_steps: int
    fusion_steps: int
    train_samples: int
    fit_samples: int


PROFILES = {
    "smoke": Profile("smoke", 16, 16, (2, 2, 2), (5, 3, 3), 2, 4, 6, 8, 400),
    "main64": Profile(
        "main64", 64, 96, (2, 2, 2), (5, 3, 3), 4, 5000, 3000, 256, 12000
    ),
    "hires96": Profile(
        "hires96", 96, 128, (3, 2, 2), (5, 3, 3), 2, 7000, 4000, 256, 16000
    ),
}

LINE_SPECS = {
    "stable_to_periodic": ("stable", "periodic"),
    "stable_to_chaotic": ("stable", "chaotic"),
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_domain(domain):
    return tuple((float(lo), float(hi)) for lo, hi in domain)


def select_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_parent_modules(fast: bool = False):
    return {
        label: (params, summarize_params(params, fast=fast))
        for label, params in HP_FALLBACK_MODULES.items()
    }


def estimate_trajectory_union_domain(
    parents, selected_labels, fast: bool = False, margin: float = 0.12
):
    trajectories = []
    for label in sorted(set(selected_labels)):
        params, _ = parents[label]
        rhs = lambda state, p=params: holling_rhs_ecology(state, p)
        if fast:
            traj, status = integrate_rhs(rhs, steps=20000, keep=6000, dt=0.05)
        else:
            traj, status = integrate_rhs(rhs, steps=80000, keep=20000, dt=0.05)
        if status == "normal" and traj.size:
            trajectories.append(traj)
    if not trajectories:
        return normalize_domain(DEFAULT_DOMAIN)

    data = np.concatenate(trajectories, axis=0)
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    span = np.maximum(hi - lo, np.array([0.05, 0.05, 0.5], dtype=np.float64))
    lo = np.maximum(0.0, lo - margin * span)
    hi = hi + margin * span
    return normalize_domain(zip(lo, hi))


def resolve_domain(domain_mode: str, parents, selected_labels, fast: bool = False):
    if domain_mode in DOMAIN_PRESETS:
        return normalize_domain(DOMAIN_PRESETS[domain_mode])
    if domain_mode == "trajectory_union":
        return estimate_trajectory_union_domain(parents, selected_labels, fast=fast)
    valid = [*sorted(DOMAIN_PRESETS), "trajectory_union"]
    raise ValueError(f"Unknown domain_mode={domain_mode}; expected one of {valid}")


def make_training_batch(params_list, profile: Profile, device, domain=DEFAULT_DOMAIN):
    batch = []
    for params in params_list:
        field = vector_field_tensor(params, profile.grid_size, domain=domain)
        x = tensor_to_torch_channels(field)
        batch.append(torch.tensor(x, dtype=torch.float32))
    return torch.stack(batch, dim=0).to(device)


def train_cae(
    profile: Profile,
    device,
    out_dir: Path,
    seed: int,
    parent_params: list[HollingParams],
    domain=DEFAULT_DOMAIN,
    checkpoint=None,
):
    torch.manual_seed(seed)
    model = EcologyCAE(
        depth=profile.depth,
        strides=profile.strides,
        kernels=profile.kernels,
    ).to(device)
    if checkpoint is not None:
        from ndtl.models import load_checkpoint

        payload = load_checkpoint(model, checkpoint, device)
        if payload.get("profile") != asdict(profile):
            raise ValueError(
                "checkpoint profile does not match the requested HP experiment"
            )
        if normalize_domain(payload.get("domain", ())) != normalize_domain(domain):
            raise ValueError(
                "checkpoint domain does not match the requested HP experiment"
            )
        if payload.get("parent_params") != [p.to_dict() for p in parent_params]:
            raise ValueError(
                "checkpoint parents do not match the requested HP experiment"
            )
        save_json(out_dir / "checkpoint_used.json", {"path": str(checkpoint)})
        return model, []
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = parent_augmented_samples(
        parent_params, seed=seed, count=profile.train_samples
    )
    losses = []
    model.train()
    for step in range(profile.train_steps):
        idx = np.random.default_rng(seed + step).choice(
            len(samples), size=profile.batch_size, replace=True
        )
        batch_params = [samples[int(i)] for i in idx]
        x = make_training_batch(batch_params, profile, device, domain=domain)
        x = x + 0.002 * torch.randn_like(x)
        opt.zero_grad(set_to_none=True)
        _, _, x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "profile": asdict(profile),
            "domain": domain,
            "parent_params": [p.to_dict() for p in parent_params],
        },
        out_dir / "cae.pt",
    )
    save_json(
        out_dir / "cae_losses.json",
        {
            "losses": losses,
            "domain": domain,
            "parent_params": [p.to_dict() for p in parent_params],
        },
    )
    return model, losses


def extract_parent_tensors(parents, profile: Profile, device, domain=DEFAULT_DOMAIN):
    out = {}
    for label, (params, summary) in parents.items():
        field = vector_field_tensor(params, profile.grid_size, domain=domain)
        x = (
            torch.tensor(tensor_to_torch_channels(field), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        out[label] = {"params": params, "summary": summary, "field": field, "tensor": x}
    return out


def content_fusion(model, x_a, x_b, alpha_b: float, steps: int, lr: float = 0.05):
    with torch.no_grad():
        feats_a, _, _ = model(x_a)
        feats_b, _, _ = model(x_b)
        init = (1.0 - alpha_b) * x_a + alpha_b * x_b
    z = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    losses = []
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        feats_z, _, _ = model(z)
        loss = 0.0
        for fa, fb, fz in zip(feats_a, feats_b, feats_z):
            loss = (
                loss
                + (1.0 - alpha_b) * F.mse_loss(fz, fa)
                + alpha_b * F.mse_loss(fz, fb)
            )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return z.detach().cpu().numpy(), losses


def evaluate_field(
    field_unit, initial_guess: HollingParams, profile: Profile, domain=DEFAULT_DOMAIN
):
    interp = TrilinearVectorField(field_unit, domain=domain)
    if profile.name == "smoke":
        traj, status = integrate_rhs(interp, steps=8000, keep=2000)
        lyap = lyapunov_spectrum(interp, transient_steps=200, measure_steps=300)
        max_nfev = 40
    else:
        traj, status = integrate_rhs(interp, steps=120000, keep=10000)
        lyap = lyapunov_spectrum(interp, transient_steps=5000, measure_steps=10000)
        max_nfev = 250
    dyn = summarize_trajectory(traj, status=status, lyapunov=lyap).to_dict()
    nonlinear = fit_holling_nonlinear(
        field_unit,
        initial=initial_guess,
        domain=domain,
        max_samples=profile.fit_samples,
        max_nfev=max_nfev,
    )
    fitted = HollingParams(**nonlinear["params"])
    basis = fit_ecological_basis_linear(
        field_unit,
        gamma_h=fitted.gamma_h,
        gamma_c=fitted.gamma_c,
        domain=domain,
        max_samples=profile.fit_samples,
    )
    return {
        "dynamics": dyn,
        "holling_fit": nonlinear,
        "basis_fit": basis,
        "trajectory": traj,
    }


def plot_trajectory(path: Path, traj: np.ndarray, title: str):
    ensure_dir(path.parent)
    fig = plt.figure(figsize=(8, 3.2), dpi=160)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    if traj.size:
        stride = max(1, traj.shape[0] // 4000)
        ax1.plot(traj[::stride, 0], traj[::stride, 1], traj[::stride, 2], lw=0.6)
    ax1.set_xlabel("P")
    ax1.set_ylabel("H")
    ax1.set_zlabel("C")
    ax2 = fig.add_subplot(1, 2, 2)
    if traj.size:
        t = np.arange(traj.shape[0])
        ax2.plot(t, traj[:, 0], label="P")
        ax2.plot(t, traj[:, 1], label="H")
        ax2.plot(t, traj[:, 2], label="C")
        ax2.legend(fontsize=7)
    ax2.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(
    profile_name: str,
    out_root: Path,
    device_arg: str = "auto",
    seed: int = 20260606,
    line: str = "both",
    domain_mode: str = "fixed02",
    checkpoint: Path | None = None,
):
    profile = PROFILES[profile_name]
    if checkpoint is not None and line == "both":
        raise ValueError(
            "select the single parent pair matching the supplied checkpoint"
        )
    out_dir = out_root / domain_mode / profile.name
    ensure_dir(out_dir)
    device = select_device(device_arg)

    parents = build_parent_modules(fast=(profile.name == "smoke"))
    if line == "both":
        lines = [(name, labels[0], labels[1]) for name, labels in LINE_SPECS.items()]
    else:
        if line not in LINE_SPECS:
            raise ValueError(
                f"Unknown line={line}; expected one of {sorted(LINE_SPECS)} or 'both'"
            )
        labels = LINE_SPECS[line]
        lines = [(line, labels[0], labels[1])]
    selected_labels = [
        label for _, a_label, b_label in lines for label in (a_label, b_label)
    ]
    domain = resolve_domain(
        domain_mode, parents, selected_labels, fast=(profile.name == "smoke")
    )

    save_json(
        out_dir / "config.json",
        {
            "profile": asdict(profile),
            "domain_mode": domain_mode,
            "domain": domain,
            "device": str(device),
        },
    )
    save_json(
        out_dir / "parents.json",
        {
            label: {"params": params.to_dict(), "summary": summary.to_dict()}
            for label, (params, summary) in parents.items()
        },
    )

    parent_tensors = extract_parent_tensors(parents, profile, device, domain=domain)

    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    report = {
        "profile": asdict(profile),
        "domain_mode": domain_mode,
        "domain": domain,
        "training_mode": "pair_specific_parent_augmented",
        "line_selection": line,
        "interpretation": "Each selected line is one two-parent ecological module assembly experiment.",
        "lines": {},
    }

    for line_name, a_label, b_label in lines:
        line_dir = out_dir / line_name
        ensure_dir(line_dir)
        pair_seed = seed + (1000 if b_label == "periodic" else 2000)
        model, losses = train_cae(
            profile,
            device=device,
            out_dir=line_dir,
            seed=pair_seed,
            parent_params=[parents[a_label][0], parents[b_label][0]],
            domain=domain,
            checkpoint=checkpoint,
        )
        x_a = parent_tensors[a_label]["tensor"]
        x_b = parent_tensors[b_label]["tensor"]
        initial_guess = parent_tensors[a_label]["params"]
        line_rows = []
        for alpha_b in alpha_values:
            fused_np, fusion_losses = content_fusion(
                model, x_a, x_b, alpha_b, steps=profile.fusion_steps
            )
            field_unit = torch_channels_to_tensor(fused_np)
            np.savez_compressed(
                line_dir / f"fused_alpha_{alpha_b:.2f}.npz",
                field_unit=field_unit,
                domain=np.asarray(domain, dtype=np.float64),
                domain_mode=domain_mode,
            )
            eval_result = evaluate_field(
                field_unit, initial_guess=initial_guess, profile=profile, domain=domain
            )
            plot_trajectory(
                line_dir / f"trajectory_alpha_{alpha_b:.2f}.png",
                eval_result["trajectory"],
                f"{line_name} alpha={alpha_b:.2f}",
            )
            row = {
                "alpha_b": float(alpha_b),
                "fusion_loss_final": fusion_losses[-1] if fusion_losses else None,
                "dynamics": eval_result["dynamics"],
                "holling_fit": eval_result["holling_fit"],
                "basis_fit": eval_result["basis_fit"],
            }
            line_rows.append(row)
            save_json(line_dir / f"eval_alpha_{alpha_b:.2f}.json", row)
        report["lines"][line_name] = {
            "cae_final_loss": losses[-1] if losses else None,
            "parent_labels": [a_label, b_label],
            "rows": line_rows,
        }
    save_json(out_dir / "report.json", report)
    return report
