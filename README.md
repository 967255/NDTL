# Neural Dynamical Transfer Learning

Code for *Dynamics creation through neural dynamical transfer learning*.
Publication: [Physical Review Research](https://doi.org/10.1103/ccgv-hgjb).

## Install

Use Python 3.9 and install the pinned scientific dependencies:

```bash
python -m pip install -r requirements.txt
python -m experiments doctor
python -m experiments check-results
```

A CUDA-enabled PyTorch installation can be used for training. CPU execution is
supported with `--device cpu` where available. The optional approximate TwoNN
mode additionally requires `hnswlib`; the default experiment settings use exact
nearest neighbors. The selected seven CAE checkpoints
are included; see [CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Layout

```text
ndtl/                       Reusable CAE, training, fields, systems, sampling, metrics
experiments/
  catalog.json              Paper experiment and source index
  cli.py                    Common launcher and run records
  classic/                  Classical-system fusion, reconstruction, evaluation
  classification/           20- and 52-system classification and clustering
  hp/                       Hastings-Powell ecology and model projection
  seirs/                    Epidemic transfer, reconstruction, and analysis
  encryption/               Chaotic image encryption
  perturbation/             Layer perturbations and feature transformations
checkpoints/                Seven selected publication CAEs
third_party/dysts/          Licensed attractor reference definitions
outputs/                    Generated runs, figures, arrays, and checkpoints (ignored)
docs/                       Experiment mapping and complete running instructions
```

Each experiment family owns its implementation and recorded small result tables.
Explicit presets and fixed metadata are under that family's `configs/` directory.
The `ndtl/` package contains shared numerical components. All new artifacts go to
`outputs/`; generated raw fields and per-pair classification checkpoints are not
included in this repository. There is no separate data download.

## Run

Run commands from the repository root. The launcher records arguments and exit
status in `run.json` and requires a new output directory for each run.

```bash
python -m experiments list
python -m experiments show E20
python -m experiments run fig5 --output outputs/classification/fig5
python -m experiments run classic-fuse --output outputs/classic/e01 -- --systems lorenz lv
python -m experiments run hp-generate --output outputs/hp/main
```

See [RUNNING.md](docs/RUNNING.md) for every experiment family and
[EXPERIMENTS.md](docs/EXPERIMENTS.md) for the paper figure/table mapping.
The source uses Python scripts with explicit inputs and outputs.

## Reference results

Small recorded metrics are kept in `experiments/<family>/results/`. Plotting jobs
can use the included classification, classic, HP, and encryption metrics. New
optimization runs may produce different values and should retain their own run
configuration. No full paper training rerun is implied by the provided checkpoints.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the DYSTS attribution and
[CITATION.cff](CITATION.cff) for citation metadata.
