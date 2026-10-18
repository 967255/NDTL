"""Paths shared by source-tree commands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
PAPER_CHECKPOINTS = {
    "hp": CHECKPOINTS / "hp/cae.pt",
    "seirs": CHECKPOINTS / "seirs/cae.pt",
    "lorenz_chen_lv": CHECKPOINTS / "lorenz_like/lorenz_chen_lv.pt",
    "lorenz_chen_lv_original": CHECKPOINTS / "lorenz_like/lorenz_chen_lv_original.pt",
    "lorenz_lv_rossler": CHECKPOINTS / "lorenz_like/lorenz_lv_rossler.pt",
    "lorenz_hadley_halvorsen": CHECKPOINTS / "lorenz_like/lorenz_hadley_halvorsen.pt",
    "lorenz_nose_hoover_rucklidge": CHECKPOINTS / "lorenz_like/lorenz_nose_hoover_rucklidge.pt",
}

def inside_project(path):
    path = Path(path)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"path must stay inside {ROOT}: {path}")
    return path
