# Publication experiments

All paper experiment implementations are organized here by scientific task:

| Directory | Entry point | Main analysis |
|---|---|---|
| `classic/` | `python -m experiments.classic.run` | `evaluate.py`, `plot.py`, `reconstruct.py` |
| `classification/` | `python -m experiments.classification.run` | Classification and clustering figure modules |
| `hp/` | `python -m experiments.hp.run` | Projection, figures, and multistart validation |
| `seirs/` | `python -m experiments.seirs.run` | `analyze.py`, `plot.py` |
| `encryption/` | `python -m experiments.encryption.run` | Image encryption metrics |
| `perturbation/` | `python -m experiments.perturbation.run` | `analyze.py`, `transformations.py` |

The common launcher is `python -m experiments`. Use `list`, `show ID`, `doctor`,
`check-results`, or `run JOB --output outputs/PATH`. `catalog.json` maps paper
figures and tables to code, presets, and recorded metrics.

See [the running guide](../docs/RUNNING.md) for all commands and input dependencies.
