"""Train the SEIRS CAE and generate the coarse/fine article sweeps."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from ndtl.models import CAE, load_checkpoint
from ndtl.training import Fusion
from ndtl.paths import ROOT
from ndtl.fields import tensor2vec, vec2tensor
from .fields import calc_grids, ode2vec, vec2ode, R
from .systems import ode_list
from .sampling import DataSampler
from .metrics import max_lyapunov_exponent, compute_power_spectrum


def fusion_verify(
    model,
    data_sampler,
    ode_test_list,
    output_dir,
    sweep,
    sys_a,
    sys_b,
    w_styl=0.01,
    cont_weight_list_1=None,
    both_styl=True,
    cont_weight_list_2=None,
    check_r0=False,
    compute_l_exp=False,
    compute_power_spec=False,
    sindy_type=None,
    grid_limits=None,
    exp_plot_save=False,
):
    SAVE_DIR = str(output_dir) + "/"
    MODEL_NAME = "seirs_" + sweep
    target_ode_list = [ode_test_list[sys_a], ode_test_list[sys_b]]
    target_tensor_list = [
        vec2tensor(ode2vec(ode, data_sampler.grids)) for ode in target_ode_list
    ]
    experiment_stamp = target_ode_list[0].name + "&" + target_ode_list[1].name
    func_save_dir = str(output_dir / experiment_stamp) + "/"
    os.makedirs(func_save_dir, exist_ok=True)
    init_image = np.random.normal(size=(*model.data_sampler.grid_shape, 3))
    init_tensor = vec2tensor(init_image)
    if cont_weight_list_1 == None:
        cont_weight_list_1 = [0.0, 0.1, 0.5, 1.0]
    if cont_weight_list_2 == None:
        cont_weight_list_2 = [0.0, 0.1, 0.5, 1.0]
    else:
        cont_weight_list_1 = cont_weight_list_1
        cont_weight_list_2 = cont_weight_list_2
    grid_shape = (48, 48, 48)
    t0 = time.time()
    vecs = np.zeros((len(cont_weight_list_1), len(cont_weight_list_2), *grid_shape, 3))
    r0s = np.zeros((len(cont_weight_list_1), len(cont_weight_list_2)))
    for i_1, w_cont_1 in enumerate(cont_weight_list_1):
        for i_2, w_cont_2 in enumerate(cont_weight_list_2):
            log_file_name = f"{w_cont_1}_{w_cont_2}_{w_styl}.log"
            with open(func_save_dir + log_file_name, "a") as log_file:
                log_file.write(
                    "loss weight: "
                    + f"w_cont_1: {w_cont_1:.2f}, w_cont_2: {w_cont_2:.2f}"
                    + "\n"
                )
            print(
                f"w_cont_1: {w_cont_1:.2f}, w_cont_2: {w_cont_2:.2f}, time = {time.time() - t0:.2f}s"
            )
            if w_cont_1 == 0 and w_cont_2 == 0:
                vec = init_image
                vecs[i_1, i_2] = vec
                if check_r0:
                    pass
                continue
            if both_styl:
                weight_list = [
                    [[w_cont_1, w_cont_1, w_cont_1], [w_styl, w_styl, w_styl]],
                    [[w_cont_2, w_cont_2, w_cont_2], [w_styl, w_styl, w_styl]],
                ]
            else:
                weight_list = [
                    [[w_cont_1, w_cont_1, w_cont_1], [0.0, 0.0, 0.0]],
                    [[w_cont_2, w_cont_2, w_cont_2], [w_styl, w_styl, w_styl]],
                ]
            (fusion_tensor, losses) = model.transfer(
                target_tensor_list,
                weight_list,
                init_tensor=init_tensor,
                save_dir=SAVE_DIR,
                verbose=False,
            )
            vec = tensor2vec(fusion_tensor)
            vecs[i_1, i_2] = vec
            if grid_limits == None:
                grid_limits = [[-25, 25], [-25, 25], [0, 50]]
            else:
                grid_limits = grid_limits
            with open(func_save_dir + log_file_name, "a") as log_file:
                log_file.write("grid limits: " + str(grid_limits) + "\n")
            grids = calc_grids(grid_limits, grid_shape)
            if sindy_type == "seris":
                ode_shifted = vec2ode(
                    vec,
                    grids=grids,
                    max_order=2,
                    extra_term_dicts={"R": R},
                    mode="linear",
                    name="fusion_shifted",
                    variables="SEI",
                )
            else:
                ode_shifted = vec2ode(
                    vec, grids=grids, max_order=2, mode="linear", name="fusion_shifted"
                )
            with open(func_save_dir + log_file_name, "a") as log_file:
                log_file.write("ode_shifted infomation: \n" + ode_shifted.info + "\n")
            if check_r0:
                pass
            if compute_l_exp:
                try:
                    lya_exp = max_lyapunov_exponent(
                        init_point=(1, 1, 1),
                        ode=ode_shifted,
                        num_points=10000,
                        omit_points=0,
                        delta0=1e-06,
                        dt=0.01,
                        dim=3,
                    )
                except IndexError as e:
                    print(f"Error occurred: {e}")
                    lya_exp = None
                if lya_exp is None:
                    with open(func_save_dir + log_file_name, "a") as log_file:
                        log_file.write("max lyapunov exp: error\n")
                else:
                    with open(func_save_dir + log_file_name, "a") as log_file:
                        log_file.write("max lyapunov exp: " + str(lya_exp) + "\n")
            if compute_power_spec:
                plot_save_dir = func_save_dir + log_file_name + "_power_spec"
                (positive_freqs, positive_power_spectrum) = compute_power_spectrum(
                    ode_shifted, init_points=(1, 1, 1), plot_save_dir=plot_save_dir
                )
            if exp_plot_save:
                plot_save_dir = func_save_dir + log_file_name + "_trajection"
                ode_shifted.plot(exp_plot_save=True, pic_save_dir=plot_save_dir)
    np.save(output_dir / "vecs.npy", vecs)
    if check_r0:
        np.save(
            SAVE_DIR
            + f"r0s_{target_ode_list[0].name}_{target_ode_list[1].name}_{MODEL_NAME}.npy",
            r0s,
        )
    print("Done!")
    return output_dir / "vecs.npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "generate"])
    parser.add_argument("--sweep", choices=["coarse", "fine"], default="coarse")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/seirs/cae.pt"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Override the selected sweep content weights.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    config = json.loads((ROOT / "experiments/seirs/configs/training.json").read_text())
    for arg, key in [
        (args.max_batches, "cae_max_batches"),
        (args.max_iters, "transfer_max_iters"),
    ]:
        if arg is not None:
            if arg < 1:
                parser.error("iteration limits must be positive")
            config[key] = arg
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / ("cae.pt" if args.action == "train" else "vecs.npy")
    if target.exists():
        parser.error(f"output already exists: {target}")
    sampler = DataSampler(config)
    sampler.add_odes([ode_list[0], ode_list[2], ode_list[4]])
    cae = CAE(
        cae_depth=config["depth"],
        cae_input_dimension=config["input_dimension"],
        cae_strides=config["strides"],
        cae_kernel_size=config["kernel_size"],
    )
    trainer = Fusion(
        cae,
        sampler,
        config=config,
        device=device,
        out_dir=args.out_dir,
        name="seirs",
        return_best=False,
    )
    sweep = json.loads(
        (ROOT / f"experiments/seirs/configs/{args.sweep}.json").read_text()
    )
    weights = args.weights or sweep["content_weights"]
    if any(not np.isfinite(value) or value < 0 for value in weights):
        parser.error("--weights must be finite and nonnegative")
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": config,
                "sweep": args.sweep,
                "content_weights": weights,
                "style_weight": sweep["style_weight"],
                "seed": args.seed,
            },
            indent=2,
        )
    )
    if args.action == "train":
        trainer.pre_train(save_name="cae.pt")
        return
    load_checkpoint(cae, args.checkpoint, device)
    fusion_verify(
        trainer,
        sampler,
        sampler.odes,
        args.out_dir,
        args.sweep,
        sys_a=2,
        sys_b=0,
        w_styl=sweep["style_weight"],
        cont_weight_list_1=weights,
        cont_weight_list_2=sweep["other_content_weights"],
        grid_limits=[[0, 1]] * 3,
        both_styl=sweep["both_style"],
    )


if __name__ == "__main__":
    main()
