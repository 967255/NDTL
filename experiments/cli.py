#!/usr/bin/env python3
"""List publication experiments and run supported commands into new output folders."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from ndtl.paths import ROOT, PAPER_CHECKPOINTS

CLASS52 = "experiments/classification/results/systems52"
CLASS20 = "experiments/classification/results/systems20"
JOBS = {
    "fig4": [
        "-m",
        "experiments.classification.plot_spectra",
        "--output-dir",
        "{output}",
    ],
    "fig5": [
        "-m",
        "experiments.classification.plot_landscape",
        "--paper-contour",
        "--contour-only",
        "--output-dir",
        "{output}",
    ],
    "fig6": [
        "-m",
        "experiments.classification.plot_strips",
        "--output-dir",
        "{output}",
    ],
    "fig18": [
        "-m",
        "experiments.classification.plot_spectral",
        "--result-dir",
        "experiments/classification/results/systems52",
        "--main-k",
        "7",
        "--dataset-label",
        "DYSTS52",
        "--file-prefix",
        "dysts52",
        "--paper-clustermap",
        "--clustermap-only",
        "--output-dir",
        "{output}",
    ],
    "fig19": [
        "-m",
        "experiments.classification.plot_spectral",
        "--main-k",
        "5",
        "--paper-embedding",
        "--embedding-only",
        "--output-dir",
        "{output}",
    ],
    "fig20": [
        "-m",
        "experiments.classification.plot_landscape",
        "--paper-surface",
        "--surface-only",
        "--output-dir",
        "{output}",
    ],
    "classic-metrics": [
        "-m",
        "experiments.classic.plot",
        "--result-dir",
        "experiments/classic/results/lorenz_lv",
        "--output-dir",
        "{output}",
    ],
    "classic-evaluate": [
        "-m",
        "experiments.classic.evaluate",
        "--experiment",
        "E01",
        "--output-dir",
        "{output}",
    ],
    "feature-transformations": [
        "-m",
        "experiments.perturbation.transformations",
        "--output-dir",
        "{output}",
    ],
    "noise-samples": ["-m", "experiments.perturbation.run", "--output-dir", "{output}"],
    "noise-analysis": [
        "-m",
        "experiments.perturbation.analyze",
        "--output-dir",
        "{output}",
    ],
    "hp-paper": ["-m", "experiments.hp.plot", "--out-dir", "{output}"],
    "hp-smoke": [
        "-m",
        "experiments.hp.run",
        "--profile",
        "smoke",
        "--line",
        "stable_to_chaotic",
        "--device",
        "cpu",
        "--out-root",
        "{output}",
    ],
    "hp-main": [
        "-m",
        "experiments.hp.run",
        "--profile",
        "main64",
        "--line",
        "stable_to_chaotic",
        "--domain-mode",
        "fixed02",
        "--out-root",
        "{output}",
    ],
    "crypto-smoke": [
        "-m",
        "experiments.encryption.run",
        "--smoke",
        "--candidate-indices",
        "6",
        "--image-names",
        "camera",
        "--image-size",
        "64",
        "--skip-parent-lyap",
        "--out-dir",
        "{output}",
    ],
    "crypto-main": ["-m", "experiments.encryption.run", "--out-dir", "{output}"],
    "classic-train": [
        "-m",
        "experiments.classic.run",
        "train",
        "--out-dir",
        "{output}",
    ],
    "classic-fuse": [
        "-m",
        "experiments.classic.run",
        "content",
        "--out-dir",
        "{output}",
    ],
    "style-scan": [
        "-m",
        "experiments.classic.run",
        "content-style",
        "--out-dir",
        "{output}",
    ],
    "cae-reconstruction": [
        "-m",
        "experiments.classic.reconstruct",
        "--output-dir",
        "{output}",
    ],
    "seirs-train": ["-m", "experiments.seirs.run", "train", "--out-dir", "{output}"],
    "seirs-coarse": [
        "-m",
        "experiments.seirs.run",
        "generate",
        "--sweep",
        "coarse",
        "--out-dir",
        "{output}",
    ],
    "seirs-fine": [
        "-m",
        "experiments.seirs.run",
        "generate",
        "--sweep",
        "fine",
        "--out-dir",
        "{output}",
    ],
    "hp-generate": [
        "-m",
        "experiments.hp.run",
        "--profile",
        "main64",
        "--line",
        "stable_to_chaotic",
        "--domain-mode",
        "fixed02",
        "--checkpoint",
        "checkpoints/hp/cae.pt",
        "--out-root",
        "{output}",
    ],
}

JOBS.update(
    {
        "classification20": [
            "-m",
            "experiments.classification.run",
            "--systems",
            "20",
            "--out-root",
            "{output}",
        ],
        "classification52": [
            "-m",
            "experiments.classification.run",
            "--systems",
            "52",
            "--out-root",
            "{output}",
        ],
        "seirs-analyze": ["-m", "experiments.seirs.analyze", "--out-dir", "{output}"],
        "seirs-plot": ["-m", "experiments.seirs.plot", "--out-dir", "{output}"],
        "hp-transients": [
            "-m",
            "experiments.hp.plot_transients",
            "--out-dir",
            "{output}",
        ],
        "hp-diagnostics": [
            "-m",
            "experiments.hp.plot_diagnostics",
            "--out-dir",
            "{output}",
        ],
    }
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def doctor():
    catalog = read_json(ROOT / "experiments/catalog.json")
    missing = []
    for experiment in catalog["experiments"]:
        for key in ("code", "results", "configs"):
            for path in experiment.get(key, []):
                if not (ROOT / path).exists():
                    missing.append({"experiment": experiment["id"], "path": path})
    dependencies = {}
    for module in [
        "numpy",
        "scipy",
        "torch",
        "matplotlib",
        "sklearn",
        "skimage",
        "PIL",
    ]:
        dependencies[module] = importlib.util.find_spec(module) is not None
    missing_weights = [
        str(p.relative_to(ROOT)) for p in PAPER_CHECKPOINTS.values() if not p.is_file()
    ]
    report = {
        "ok": not missing and not missing_weights and all(dependencies.values()),
        "retained_checkpoints": {
            "expected": len(PAPER_CHECKPOINTS),
            "missing": missing_weights,
        },
        "source_and_reference_paths_missing": missing,
        "dependencies": dependencies,
        "note": "Selected HP, classic-system and SEIRS CAEs are included. Raw fields and classification checkpoints must be generated.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("experiment")
    sub.add_parser("doctor")
    sub.add_parser("check-results")
    run = sub.add_parser("run")
    run.add_argument("job", choices=sorted(JOBS))
    run.add_argument("--output")
    run.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.action != "run" and extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if args.action == "check-results":
        from experiments.validate_results import main as check_results

        check_results()
        return 0
    if args.action == "doctor":
        return doctor()
    if args.action in ("list", "show"):
        entries = read_json(ROOT / "experiments/catalog.json")["experiments"]
        if args.action == "list":
            for item in entries:
                print("{}  {}  {}".format(item["id"], item["name"], item["paper"]))
        else:
            found = [item for item in entries if item["id"] == args.experiment]
            if len(found) != 1:
                parser.error("unknown experiment ID")
            print(json.dumps(found[0], ensure_ascii=False, indent=2))
        return 0
    output_flags = {"--out-dir", "--out-root", "--output-dir"}
    if any(arg.split("=")[0] in output_flags for arg in extra):
        parser.error("select the output folder with --output")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        ROOT / (args.output or "outputs/{}/{}".format(args.job, timestamp))
    ).resolve()
    if (ROOT / "outputs").resolve() not in output.parents:
        parser.error("--output must be a new directory under outputs/")
    command = (
        [sys.executable]
        + [arg.replace("{output}", str(output)) for arg in JOBS[args.job]]
        + extra
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if output.exists():
        parser.error("output already exists; choose another directory")
    output.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/xdg"))
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env.setdefault(name, "1")
    record = {"job": args.job, "command": command, "started_utc": timestamp}
    log = output / "run.json"
    log.write_text(json.dumps(record, indent=2))
    result = subprocess.run(command, cwd=ROOT, env=env)
    record.update(
        exit_code=result.returncode,
        finished_utc=datetime.datetime.utcnow().isoformat() + "Z",
    )
    log.write_text(json.dumps(record, indent=2))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
