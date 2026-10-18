# Classification and clustering

`run.py --systems 20|52` selects the benchmark. `train20.py` provides the shared pair-specific CAE/fusion pipeline; `train52.py` extends system selection and merges the 20-system block. `arguments.py` and `plot_utils.py` hold shared options and plotting primitives. `configs/` holds fixed attractor boxes; `results/systems20/` and `results/systems52/` hold recorded primary tables.

See [RUNNING.md](../../docs/RUNNING.md) for commands. Write new artifacts under
`outputs/classification/`; `results/` directories contain recorded reference metrics.
