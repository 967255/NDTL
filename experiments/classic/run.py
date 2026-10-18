"""Train classic-system CAEs and generate content or content/style scans."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ndtl import systems
from ndtl.fields import ensemble_rescaling, ode2vec, tensor2vec, vec2tensor
from ndtl.models import CAE, load_checkpoint
from ndtl.paths import ROOT
from ndtl.sampling import DataSampler
from ndtl.training import Fusion

DEFAULT_CONTENT_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_STYLE_WEIGHTS = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]


def build_weights(content_ratio, style_ratio, content_strength, style_strength, layers):
    return [
        [
            [content_strength * content_ratio / layers] * layers,
            [style_strength * style_ratio / layers] * layers,
        ],
        [
            [content_strength * (1.0 - content_ratio) / layers] * layers,
            [style_strength * (1.0 - style_ratio) / layers] * layers,
        ],
    ]


def build_experiment(preset_name, device, output, max_batches=None, max_iters=None):
    preset = json.loads(
        (ROOT / "experiments/classic/configs/presets.json").read_text()
    )[preset_name]
    config = json.loads(
        (ROOT / "experiments/classic/configs/training.json").read_text()
    )
    config["grid_limits"] = preset["grid_limits"]
    if max_batches is not None:
        config["cae_max_batches"] = max_batches
    if max_iters is not None:
        config["transfer_max_iters"] = max_iters
    sources = [getattr(systems, name) for name in preset["systems"]]
    if preset["normalized"]:
        sources = [ensemble_rescaling(ode)[1] for ode in sources]
    sampler = DataSampler(
        grid_shape=config["grid_shape"],
        grid_limits=config["grid_limits"],
        turb_scale=config["turb_scale"],
    )
    sampler.add_odes(sources)
    cae = CAE(
        cae_depth=config["depth"],
        cae_input_dimension=config["input_dimension"],
        cae_strides=config["strides"],
        cae_kernel_size=config["kernel_size"],
    )
    trainer = Fusion(cae, sampler, config=config, device=device, out_dir=output)
    return preset, sources, trainer


def generate(trainer, sources, pair, weights, mode, output):
    """Use one shared noise field across a scan, preserving the classic protocol."""
    config = trainer.config
    tensors = [
        vec2tensor(ode2vec(sources[i], trainer.data_sampler.grids)) for i in pair
    ]
    shape = tuple(trainer.data_sampler.grid_shape)
    initial_field = np.random.normal(size=(*shape, 3)).astype(np.float32)
    initial_tensor = vec2tensor(initial_field)
    vectors = np.zeros((len(weights), 1, *shape, 3), dtype=initial_field.dtype)
    history = []
    for i, weight in enumerate(weights):
        if mode == "content":
            content_ratio, style_ratio, content_strength, style_strength = (
                weight,
                0.0,
                1.0,
                0.0,
            )
        else:
            content_ratio, style_ratio = 1.0, 0.0
            content_strength, style_strength = weight, 1.0 - weight
        target_weights = build_weights(
            content_ratio,
            style_ratio,
            content_strength,
            style_strength,
            trainer.cae.n_layers,
        )
        vector, losses = trainer.transfer(
            tensors,
            target_weights,
            init_tensor=initial_tensor,
            save_dir=str(output) + "/",
            verbose=True,
        )
        vectors[i, 0] = tensor2vec(vector)
        history.append({"weight": weight, "iterations": len(losses)})
        np.save(output / f"losses_{i:02d}.npy", losses)
    target = output / "vecs.npy"
    np.save(target, vectors)
    (output / "scan.json").write_text(
        json.dumps(
            {"mode": mode, "weights": weights, "history": history, "config": config},
            indent=2,
        )
    )
    return target


def main():
    presets = json.loads(
        (ROOT / "experiments/classic/configs/presets.json").read_text()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "content", "content-style"])
    parser.add_argument("--preset", choices=sorted(presets), default="lorenz_chen_lv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--systems", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"))
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    for name in ["max_batches", "max_iters"]:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / ("cae.pt" if args.action == "train" else "vecs.npy")
    if target.exists():
        parser.error(f"output already exists: {target}")
    preset, sources, trainer = build_experiment(
        args.preset, device, args.out_dir, args.max_batches, args.max_iters
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {"preset": args.preset, "config": trainer.config, "seed": args.seed},
            indent=2,
        )
    )
    if args.action == "train":
        trainer.pre_train(save_name="cae.pt")
        return
    checkpoint = args.checkpoint or ROOT / preset["checkpoint"]
    load_checkpoint(trainer.cae, checkpoint, device)
    names = args.systems or preset["systems"][:2]
    if any(name not in preset["systems"] for name in names):
        parser.error("--systems must be selected from the preset training systems")
    pair = [preset["systems"].index(name) for name in names]
    weights = args.weights or (
        DEFAULT_CONTENT_WEIGHTS if args.action == "content" else DEFAULT_STYLE_WEIGHTS
    )
    if any(not np.isfinite(w) or w < 0.0 or w > 1.0 for w in weights):
        parser.error("--weights must be finite and between 0 and 1")
    print(generate(trainer, sources, pair, weights, args.action, args.out_dir))


if __name__ == "__main__":
    main()
