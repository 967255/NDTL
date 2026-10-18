# Recorded results and generated inputs

Small recorded result tables are colocated with their experiment implementations
under `experiments/<family>/results/`. The seven selected CAEs are listed in
[CHECKPOINTS.md](CHECKPOINTS.md). Fixed classification grid metadata and explicit
classic/SEIRS presets are under each experiment's `configs/` directory.

Raw vector fields, per-pair classification checkpoints, perturbation sample
collections, and generated display summaries are not distributed. Generate the
inputs with the commands in [RUNNING.md](RUNNING.md). No separate data download
is required or provided. New outputs, including newly generated arrays and
figures, go under the Git-ignored `outputs/` directory.

The SEIRS directory contains generation and analysis code plus configuration;
its field arrays and analysis tables are generated locally. It has no recorded
result directory containing only configuration files.

Reference numeric values are preserved. Recorded configurations describe completed
runs, including worker shards where applicable; merged tables cover the full
experiment. New runs retain their own configurations and may produce different
numeric results.
