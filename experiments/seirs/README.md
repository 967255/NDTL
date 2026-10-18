# SEIRS epidemic transfer

`run.py` trains the CAE or generates coarse/fine fields using `configs/`. `polynomial.py` defines the shared quadratic basis and sparse fitting model. `analyze.py` exports fitted coefficients, reproduction ratios, and trajectories. `plot.py` renders those tables. `fields.py`, `systems.py`, `sampling.py`, and `metrics.py` retain the SEIRS-specific numerical conventions.

See [RUNNING.md](../../docs/RUNNING.md#seirs-epidemic-transfer) for the complete
generate, analyze, plot, and fit pipeline. New artifacts go under `outputs/seirs/`.
