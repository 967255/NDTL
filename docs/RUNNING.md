# Running the experiments

Run all commands from the repository root after installing `requirements.txt`.
Use a fresh path under `outputs/` for `--output`. Add `--dry-run` to a launcher
command to inspect it without execution; put experiment arguments after `--`.
For module-specific help, use, for example,
`python -m experiments.classic.run --help`.

## Classical-system fusion

Presets and optimizer settings are in `experiments/classic/configs/`.
The included CAEs support generation without repeating CAE training. The default
preset pair is Lorenz-Chen; the E01 command selects Lorenz-Lv explicitly.

```bash
python -m experiments run classic-fuse --output outputs/classic/e01 -- --systems lorenz lv
python -m experiments run classic-evaluate --output outputs/classic/e01_evaluation -- --experiment E01 --vec-path outputs/classic/e01/vecs.npy
python -m experiments run classic-metrics --output outputs/classic/e01_figures
python -m experiments run cae-reconstruction --output outputs/classic/reconstruction
```

`classic-metrics` uses the included E01 metrics by default. To plot regenerated
metrics, pass `--result-dir outputs/classic/e01_evaluation` after `--`. Add
`--trajectories` to plot the reconstructed equations as well.

Other fusion pairs use the following preset and system arguments:

| ID | `--preset` | `--systems` | Recorded metrics directory under `experiments/classic/results/` |
|---|---|---|---|
| E01 | `lorenz_chen_lv` | `lorenz lv` | `lorenz_lv` |
| E02 | `lorenz_chen_lv` | `lorenz chen` | `lorenz_chen` |
| E03 | `lorenz_hadley_halvorsen` | `lorenz halvorsen` | `lorenz_halvorsen` |
| E04 | `lorenz_nose_hoover_rucklidge` | `lorenz nose_hoover` | `lorenz_nose_hoover` |
| E05 | `lorenz_lv_rossler` | `lv rossler` | `lv_rossler` |
| E06 | `lorenz_hadley_halvorsen` | `hadley halvorsen` | `hadley_halvorsen` |
| E07 | `lorenz_nose_hoover_rucklidge` | `nose_hoover rucklidge` | `nose_hoover_rucklidge` |

For example:

```bash
python -m experiments run classic-fuse --output outputs/classic/e03 -- --preset lorenz_hadley_halvorsen --systems lorenz halvorsen
python -m experiments run classic-evaluate --output outputs/classic/e03_evaluation -- --experiment E03 --vec-path outputs/classic/e03/vecs.npy
python -m experiments run style-scan --output outputs/classic/content_style
```

The content/style scan is E08. To retrain a preset CAE, use `classic-train` with
`--preset`; this writes a new checkpoint under the selected output directory.
To test execution with a short optimization, add `--max-iters 1 --weights 0.5
--device cpu`. Such a run is only an execution check.

## Classification and clustering

The 20- and 52-system pipelines share CAE training, fusion, distance construction,
CLI options, and plotting utilities. Fixed attractor boxes are in
`experiments/classification/configs/`. Reference tables are in
`experiments/classification/results/systems20/` and `systems52/`.

```bash
python -m experiments run classification20 --output outputs/classification/systems20
python -m experiments run classification52 --output outputs/classification/systems52
```

The 52-system run reuses the included 20-system block by default. To reuse a newly
computed block, pass `--reference-results-dir outputs/classification/systems20`.
Use `--pair-shard-index` and `--pair-shard-count` for distributed runs, then
`--merge-only` with the same output path through the direct module command.
The launcher reserves fresh paths; resumable worker and merge operations should
use `python -m experiments.classification.run --systems 52 --out-root PATH ...`.

Generate figures from the included tables without retraining:

```bash
python -m experiments run fig4  --output outputs/classification/fig4
python -m experiments run fig5  --output outputs/classification/fig5
python -m experiments run fig6  --output outputs/classification/fig6
python -m experiments run fig18 --output outputs/classification/fig18
python -m experiments run fig19 --output outputs/classification/fig19
python -m experiments run fig20 --output outputs/classification/fig20
```

Generated display summaries are saved with the figures and are not distributed.
The `--smoke-test` option reduces the classification workload; explicit optimizer
and Lyapunov step arguments can reduce it further for an execution check.

## Hastings-Powell ecology

```bash
python -m experiments run hp-generate --output outputs/hp/main
python -m experiments run hp-paper --output outputs/hp/figures
python -m experiments run hp-transients --output outputs/hp/transients
python -m experiments run hp-diagnostics --output outputs/hp/diagnostics
```

`hp-generate` reuses `checkpoints/hp/cae.pt` for the retained main profile/domain
and stable-to-chaotic parent pair. `hp-main` repeats CAE training; `hp-smoke` uses
a reduced profile. `hp-paper` and `hp-transients` use `experiments/hp/results/main/report.json` by
default. `hp-diagnostics` uses the newly generated
`outputs/hp/main/fixed02/main64/report.json` and its raw field files. To choose
a different report, pass `--report` with its path.

The multistart check needs raw generated fields next to its report:

```bash
python -m experiments.hp.validate_multistart --report outputs/hp/main/fixed02/main64/report.json --out outputs/hp/multistart.json
```

See [HP.md](HP.md) for the model projection and report contents.

## SEIRS epidemic transfer

The coarse/fine weight scans and training configuration are in
`experiments/seirs/configs/`. The following pipeline replaces interactive
session state with explicit inputs and produces the trajectory, coefficient,
and reproduction-ratio plots:

```bash
python -m experiments run seirs-coarse --output outputs/seirs/coarse
python -m experiments run seirs-fine --output outputs/seirs/fine
python -m experiments run seirs-analyze --output outputs/seirs/analysis
python -m experiments run seirs-plot --output outputs/seirs/figures
```

`seirs-analyze` reads the two `vecs.npy` files, fits the shared quadratic polynomial
basis with LASSO (`alpha=0.005`), and exports coefficient tables, reproduction
ratios, and selected trajectories. Default trajectory indices are coarse 6 and
fine 9. `seirs-plot` reads those tables and plots fine-sweep coefficient heatmaps
for indices 0, 4, and 8. Input paths and trajectory indices can be selected
through CLI flags.

For custom generated paths, pass `--coarse PATH --fine PATH` to `seirs-analyze`,
and `--analysis-dir PATH` to `seirs-plot`.
`seirs-train` retrains the CAE. The analysis code validates the field shape and
reads generated weight metadata when available.

## Chaotic image encryption

```bash
python -m experiments run style-scan --output outputs/encryption/fields
python -m experiments run crypto-smoke --output outputs/encryption/smoke -- --vec-path outputs/encryption/fields/vecs.npy
python -m experiments run crypto-main --output outputs/encryption/main -- --vec-path outputs/encryption/fields/vecs.npy
```

The default article candidate is index 6 of the nine-point content/style scan.
See [ENCRYPTION.md](ENCRYPTION.md).

## Feature transformations and layer perturbations

```bash
python -m experiments run feature-transformations --output outputs/perturbation/transformations
python -m experiments run noise-samples --output outputs/perturbation/samples
python -m experiments run noise-analysis --output outputs/perturbation/analysis -- --run-dir outputs/perturbation/samples/lorenz
```

The perturbation generator uses the original-coordinate CAE. The recorded small
metrics in `experiments/perturbation/results/` support inspection; raw sample
fields must be generated for a new analysis.
