#!/usr/bin/env python3
"""Build website application assets from published results without training NDTL.

Requires numpy, scipy, matplotlib; PDF extraction uses pypdf. In this workspace:
  MPLCONFIGDIR=/tmp/ndtl-mpl /opt/anaconda3/bin/python3 tools/prepare_applications.py \
    --pdf-python /Users/mahe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

The HP trajectories below are new numerical integrations of the published model
parameters, not the raw optimized NDTL tensor. Every output retains provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.integrate import solve_ivp

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
BG, TEXT, MUTED, RULE = "#080b10", "#edece8", "#aaadb5", "#292d34"
COLORS = ["#a3b2c6", "#b4a1c7", "#dbb57a"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": RULE,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": RULE,
        "grid.alpha": .7, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4,
        "savefig.bbox": "tight", "savefig.pad_inches": .08,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save_figure(fig, out):
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(out.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def extract_published_figures(pdf, out, pdf_python):
    # Exact raster streams embedded in the published PDF, with no recoloring.
    program = r'''
from pypdf import PdfReader
from pathlib import Path
import sys
r = PdfReader(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
for page, image_name, output in [(7,"Im8.png","fig7-hp-original.png"),(8,"Im9.png","fig8-seirs-original.png"),(8,"Im10.png","fig9-encryption-original.png")]:
    images={im.name:im for im in r.pages[page].images}
    (out/output).write_bytes(images[image_name].data)
'''
    subprocess.run([str(pdf_python), "-c", program, str(pdf), str(out)], check=True)
    shutil.copy2(pdf, out / "ndtl-prr-2026.pdf")


def hp_rhs(_, state, p):
    P, H, C = state
    uh = p["alpha_h"] * p["beta_h"] * P / (p["gamma_h"] + P)
    uc = p["alpha_c"] * p["beta_c"] * H / (p["gamma_c"] + H)
    return [P * (1-P/p["K"]) - uh*H,
            H * (uh-p["d_h"]) - uc*C,
            C * (uc-p["d_c"])]


def prepare_hp(repo, data, figures):
    sub = Path("results_display/hp_ecology_ndtl/fixed02/main64")
    parents = load(repo / sub / "parents.json")
    evaluation = load(repo / sub / "stable_to_chaotic/eval_alpha_0.50.json")
    floquet = load(repo / "results_display/hp_ecology_ndtl/validation/fixed02_stable_to_chaotic_alpha0p50_holling_fit_floquet.json")
    systems = [
        ("hp-stable", "Stable parent", "Coexistence equilibrium", parents["stable"]["params"]),
        ("hp-chaotic", "Chaotic parent", "Chaotic coexistence", parents["chaotic"]["params"]),
        ("hp-projected", "Projected NDTL system", "Oscillatory coexistence", evaluation["holling_fit"]["params"]),
    ]
    short_t = np.linspace(0, 600, 1200)
    tail_t = np.linspace(3000, 4000, 1200)
    ts = np.unique(np.concatenate([short_t, tail_t]))
    result = []
    fig = plt.figure(figsize=(10.0, 5.5), layout="constrained")
    for i, (slug, title, behavior, p) in enumerate(systems):
        sol = solve_ivp(hp_rhs, [0, 4000], [.5, .2, 8], args=(p,),
                        t_eval=ts, method="DOP853", rtol=2e-10, atol=1e-12, max_step=.2)
        if not sol.success or not np.isfinite(sol.y).all() or (sol.y < 0).any():
            raise RuntimeError("Invalid HP integration: " + title)
        short = sol.y[:, :1200].T
        tail = sol.y[:, -1200:].T
        result.append({
            "id": slug, "name": title, "behavior": behavior, "color": COLORS[i],
            "params": p, "time": np.round(short_t, 5).tolist(),
            "trajectory": np.round(short, 7).tolist(),
            "tailTime": np.round(tail_t, 5).tolist(),
            "tailTrajectory": np.round(tail, 7).tolist(),
            "minTail": tail.min(axis=0).tolist(),
        })
        ax = fig.add_subplot(2, 3, i+1, projection="3d")
        ax.set_facecolor(BG)
        ax.plot(short[:, 0], short[:, 1], short[:, 2], color=COLORS[i], lw=.7, alpha=.8)
        if i == 2:
            ax.plot(tail[-240:, 0], tail[-240:, 1], tail[-240:, 2], color=COLORS[i], lw=1.6)
        ax.set_title(title, color=COLORS[i], pad=10)
        ax.set_xlabel("P", labelpad=-2); ax.set_ylabel("H", labelpad=-2); ax.set_zlabel("C", labelpad=-2)
        ax.view_init(24, -58)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor(RULE)
            axis._axinfo["grid"]["color"] = RULE
            axis._axinfo["grid"]["linewidth"] = .5
            axis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8, pad=-1)
        ax2 = fig.add_subplot(2, 3, i+4)
        ax2.plot(short_t, short[:, 2], color=COLORS[i], lw=1.25)
        ax2.set_xlabel("Time"); ax2.set_ylabel("Carnivore C")
        ax2.set_xlim(0, 600); ax2.set_ylim(7.2, 14.2)
        ax2.grid(axis="y")
        ax2.text(0, 1.05, behavior, transform=ax2.transAxes, color=MUTED, fontsize=9)
    save_figure(fig, figures / "hp-coexistence")
    write(data / "hp-trajectories.json", {
        "schemaVersion": 1,
        "provenance": {"kind": "published-model-integration", "source": str(sub),
                       "paper": "Fig. 7; Appendix F", "note": "New integration of the exact saved HP parent and projected-model parameters. Not raw optimized tensor trajectories."},
        "axes": ["P", "H", "C"],
        "integration": {"method": "DOP853", "initialState": [.5, .2, 8], "maxStep": .2,
                        "rtol": 2e-10, "atol": 1e-12, "interval": [0, 4000],
                        "displayInterval": [0, 600], "tailInterval": [3000, 4000], "pointsPerSegment": 1200},
        "systems": result,
    })
    return evaluation, floquet


TRANSFORMATIONS = [
    ("Lorenz", "Time reversal", 4.000, 0.000),
    ("Lorenz", "Coordinate exchange", 1.033, 2.76e-31),
    ("Lorenz", "Reflection", 12.543, .115),
    ("Lorenz", "Rotation 30 degrees", 1.734, .119),
    ("Chen", "Time reversal", 4.000, 0.000),
    ("Chen", "Coordinate exchange", 3.670, 4.61e-31),
    ("Chen", "Reflection", 8.804, .695),
    ("Chen", "Rotation 30 degrees", 1.852, .270),
    ("Lv", "Time reversal", 4.000, 0.000),
    ("Lv", "Coordinate exchange", 3.588, 4.16e-30),
    ("Lv", "Reflection", 8.223, .654),
    ("Lv", "Rotation 30 degrees", 1.932, .275),
    ("Rössler", "Time reversal", 4.000, 0.000),
    ("Rössler", "Coordinate exchange", 1.073, 7.42e-26),
    ("Rössler", "Reflection", 7.989, .258),
    ("Rössler", "Rotation 30 degrees", .336, .0856),
]


def prepare_features(repo, data, figures):
    source = "results_lirt/LIRT_pre_noise_lorenz_chen_lv_ori/lorenz/eval_lorenz.json"
    report = load(repo / source)
    metric_keys = ["sigma_hat", "rho_hat", "beta_hat", "D_hat", "lambda1_hat", "lambda2_hat", "lambda3_hat", "ky_dim_hat", "fit_rel_rmse_total"]
    summaries = [{k: v for k, v in row.items() if k in ["layer_id", "epsilon_nominal", "n", "lyap_success_n"] + metric_keys}
                 for row in report["grouped_summary_layer_epsilon"]]
    raw = [{k: row[k] for k in ["layer_id", "epsilon_nominal", "seed_id"] + metric_keys}
           for row in report["rows_noisy"]]
    write(data / "feature-analysis.json", {
        "schemaVersion": 1, "source": source, "sourceSha256": digest(repo / source),
        "paper": "Fig. 3 and Table II", "layerwiseProtocol": report["protocol"],
        "baseline": report["baseline_clean_reconstruction_fit"],
        "originalBaseline": report["baseline_original_vector_field_fit"],
        "summaries": summaries, "rows": raw,
        "transformations": [{"system": s, "transformation": t, "deltaPoint": p, "deltaGram": g}
                            for s,t,p,g in TRANSFORMATIONS],
        "transformationProvenance": "Manually transcribed from published Table II, Appendix B; dimensionless discrepancies. Coordinate exchange includes channel alignment.",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), layout="constrained")
    specifications = [("lambda1_hat", "Maximal Lyapunov exponent"), ("ky_dim_hat", "Kaplan–Yorke dimension"), ("fit_rel_rmse_total", "Lorenz-family regression residual")]
    for ax, (key, title) in zip(axes, specifications):
        for layer in [0,1,2]:
            rows = sorted((r for r in summaries if r["layer_id"]==layer), key=lambda r:r["epsilon_nominal"])
            x = np.array([r["epsilon_nominal"] for r in rows]); y = np.array([r[key]["mean"] for r in rows])
            low = np.array([r[key]["q25"] for r in rows]); high = np.array([r[key]["q75"] for r in rows])
            ax.plot(x, y, color=COLORS[layer], label=f"Layer {layer}", marker=["o","s","^"][layer], markersize=3)
            ax.fill_between(x, low, high, color=COLORS[layer], alpha=.10)
        ax.axhline(report["baseline_clean_reconstruction_fit"][key], color=MUTED, ls=":", lw=.8)
        ax.set_xscale("log"); ax.set_xlabel("Relative Gaussian noise, ε")
        ax.set_title(title, fontsize=10); ax.grid(axis="y")
    axes[0].legend(fontsize=8)
    save_figure(fig, figures / "layerwise-perturbation")

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.25), sharey=True, layout="constrained")
    names = ["Time reversal", "Coordinate exchange", "Reflection", "Rotation 30 degrees"]
    for ax, name in zip(axes, names):
        rows = [r for r in TRANSFORMATIONS if r[1] == name]
        pos = np.arange(4)
        for j, (_, _, point, gram) in enumerate(rows):
            ax.plot([gram,point], [j,j], color=RULE, lw=1.8)
        ax.scatter([r[2] for r in rows], pos, c=COLORS[0], label="Pointwise content", s=28)
        ax.scatter([r[3] for r in rows], pos, c=COLORS[2], marker="D", label="Gram style", s=25)
        ax.set_yticks(pos, [r[0] for r in rows])
        ax.set_xlim(-.2, max(r[2] for r in rows)*1.12)
        ax.set_title(name.replace(" degrees", "°"), fontsize=10)
        ax.set_xlabel("Normalized discrepancy"); ax.grid(axis="x")
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    save_figure(fig, figures / "content-style-transformations")
    return {"dataFile": "data/feature-analysis.json", "source": source, "numSeeds": 10,
            "numRawRows": len(raw), "numSummaries": len(summaries), "numTransformations": 16,
            "layerFigure": "assets/figures/layerwise-perturbation.svg",
            "transformationFigure": "assets/figures/content-style-transformations.svg"}


def prepare_crypto(repo, figures):
    sub = Path("results_display/crypto_lorenz_chen_fusion")
    with (repo / sub / "source_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    selected = [next(r for r in rows if r["source_name"]==name)
                for name in ["lorenz_unit_nv", "chen_unit_nv", "fusion_vec_6"]]
    keys = ["cipher_entropy_mean", "adj_corr_abs_mean_mean", "npcr_mean", "uaci_mean", "key_npcr_mean", "key_uaci_mean", "crypto_score_mean"]
    benchmark = [{"id": row["source_name"], "label": label, "color": color,
                  **{k: float(row[k]) for k in keys},
                  **{k.rsplit("_",1)[0]+"_std": float(row[k.rsplit("_",1)[0]+"_std"]) for k in keys}}
                 for row, label, color in zip(selected, ["Lorenz", "Chen", "NDTL fused"], COLORS)]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.6), layout="constrained")
    for ax, key, label in zip(axes, keys[:3], ["Cipher entropy (bits)", "Absolute adjacent correlation", "NPCR (%)"]):
        for y, row in enumerate(benchmark):
            std = row[key.rsplit("_",1)[0]+"_std"]
            ax.errorbar(row[key], y, xerr=std, color=row["color"], fmt="o", markersize=5, capsize=3, lw=1)
        ax.set_yticks(range(3), [r["label"] for r in benchmark]); ax.invert_yaxis()
        ax.set_xlabel(label); ax.grid(axis="x")
        ax.ticklabel_format(useOffset=False, axis="x")
        ax.xaxis.set_major_locator(MaxNLocator(3))
    fig.suptitle("Published benchmark: mean ± SD across three images", fontsize=9, color=MUTED)
    save_figure(fig, figures / "encryption-benchmark")
    return benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT / "GitHub_更新预览_20261017/NDTL")
    parser.add_argument("--pdf", type=Path, default=ROOT / "Dynamics creation through neural dynamical transfer learning.pdf")
    parser.add_argument("--pdf-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    data, figures, paper = BASE/"dist/data", BASE/"dist/assets/figures", BASE/"dist/assets/paper"
    for path in [data, figures, paper]: path.mkdir(parents=True, exist_ok=True)
    theme()
    extract_published_figures(args.pdf, paper, args.pdf_python)
    hp, floquet = prepare_hp(args.repo, data, figures)
    features = prepare_features(args.repo, data, figures)
    benchmark = prepare_crypto(args.repo, figures)
    result = {
        "meta": {"schemaVersion": 1, "paperTitle": "Dynamics creation through neural dynamical transfer learning",
                 "authors": "He Ma, Qiyang Ge, Yu Meng, Celso Grebogi, and Wei Lin",
                 "citation": "Physical Review Research 8, 033337 (2026)",
                 "doi": "https://doi.org/10.1103/ccgv-hgjb", "paperFile": "assets/paper/ndtl-prr-2026.pdf",
                 "license": "CC BY 4.0", "sourcePdfSha256": digest(args.pdf)},
        "hp": {
            "title": "Creating oscillatory ecological coexistence",
            "summary": "A stable food chain and a chaotic food chain yield a projected HP model with persistent, periodic coexistence of resource, herbivore, and carnivore.",
            "sourceFigure": "Fig. 7", "sourceAppendix": "Appendix F",
            "originalFigure": "assets/paper/fig7-hp-original.png", "figure": "assets/figures/hp-coexistence.svg",
            "trajectoryFile": "data/hp-trajectories.json", "methodAnchor": "#method-regression",
            "fusionMode": "content + content", "alphaB": .5,
            "equations": ["dP/dt = P(1 − P/K) − u_h(P)H", "dH/dt = H[u_h(P) − d_h] − u_c(H)C", "dC/dt = C[u_c(H) − d_c]", "u_h(P) = α_h P/(γ_h + P)", "u_c(H) = α_c H/(γ_c + H)"],
            "projectedParameters": hp["holling_fit"]["params"],
            "projectionRelativeRmse": hp["holling_fit"]["rel_rmse"],
            "validation": {"projectedPersistent": "27/27", "projectedOscillatory": "27/27", "rawTensorPersistent": "23/27", "rawTensorOscillatory": "7/27", "floquetPeriod": floquet["period"], "floquetMagnitudes": floquet["abs_floquet_multipliers"]},
            "qualification": "The ecological conclusion concerns the projection into the HP model family. The raw optimized tensor has different persistence and oscillation outcomes.",
            "provenance": ["results_display/hp_ecology_ndtl/fixed02/main64/parents.json", "results_display/hp_ecology_ndtl/fixed02/main64/stable_to_chaotic/eval_alpha_0.50.json", "results_display/hp_ecology_ndtl/validation/fixed02_stable_to_chaotic_alpha0p50_holling_fit_floquet.json"],
        },
        "seirs": {
            "title": "Interpretable epidemiological models",
            "summary": "SEIRS content and Lorenz style produce sparse equations with additional exposed-compartment inflow and nonlinear infection terms, within a range of feature weights.",
            "sourceFigure": "Fig. 8", "sourceEquations": "Eqs. (1)–(2)",
            "originalFigure": "assets/paper/fig8-seirs-original.png", "methodAnchor": "#method-style",
            "fusionMode": "SEIRS content + Lorenz style", "regression": "LASSO, 0.005",
            "initialStateSEI": [.99, 0, .01], "populationConstraint": "S + E + I + R = 1",
            "equations": ["dS/dt = f(S,E,I,R)", "dE/dt = κS − σ₁E + βSI", "dI/dt = σ₂E − γI + ξE²", "dR/dt = −dS/dt − dE/dt − dI/dt"],
            "representativePublishedEquation": {"contentStyleRatio": 1e5, "coefficientPrecision": "rounded as printed in Fig. 8(c)", "equations": ["dS/dt = −0.04S − 0.04E − 0.04I − 0.87SI", "dE/dt = 0.01S − 0.04E + 0.73SI", "dI/dt = 0.01E − 0.14I + 0.01E²", "dR/dt = 0.03S + 0.07E + 0.18I + 0.14SI − 0.01E²"]},
            "qualification": "β/γ measures the fitted transmission–recovery balance; it is not the overall basic reproduction number of the new equations. Very low content/style ratios can lose the original SEIRS characteristics.",
            "dataAvailability": "Published image and saved experiment configurations are available locally. Numerical field sweeps are listed in the companion manifest but are not present in this code preview; no synthetic trajectory is substituted.",
            "provenance": ["Published Fig. 8 and Section V B", "experiments/seirs/README.md", "experiments/seirs/results/exp_lls_styleon/config_transfer_exp_lls_styleon.json"],
        },
        "encryption": {
            "title": "Generated chaos as an algorithmic source",
            "summary": "A Lorenz-content/Chen-style system drives a permutation–diffusion pipeline, producing noise-like ciphertext and exact recovery with the correct key in the published demonstration.",
            "sourceFigure": "Fig. 9", "sourceTable": "Table VI", "sourceAppendix": "Appendix G",
            "originalFigure": "assets/paper/fig9-encryption-original.png", "figure": "assets/figures/encryption-benchmark.svg",
            "methodAnchor": "#method", "selectedSource": "fusion_vec_6", "benchmark": benchmark,
            "protocol": {"images": ["camera", "moon", "astronaut"], "imageSize": [512,512], "initialState": [.3,.3,.5], "dt": .01, "burnInSteps": 1000, "keyPerturbation": 1e-14, "diffusionPasses": 2},
            "qualification": "This is the published downstream demonstration and benchmark, not a claim of comprehensive cryptographic security or uniform superiority on every metric.",
            "provenance": ["results_display/crypto_lorenz_chen_fusion/source_summary.csv", "results_display/crypto_lorenz_chen_fusion/winner_report.json"],
        },
        "featureAnalysis": {**features,
            "contentSummary": "Spatially resolved activation tensors retain local or aggregated vector-field structure, depending on encoder depth.",
            "styleSummary": "Gram matrices summarize second-order feature-channel correlations while contracting the spatial index.",
            "qualification": "Input-layer time reversal and aligned coordinate exchange preserve Gram features. Reflection and rotation are only comparatively stable in the tested systems; there is no universal rotation invariance or general energy interpretation.",
        },
    }
    write(data / "applications.json", result)
    print(json.dumps({"output": str(data/"applications.json"), "hpSystems": 3, "hpPointsPerSegment": 1200, "featureRows": features["numRawRows"], "transformations": 16, "figures": 4}))


if __name__ == "__main__":
    main()
