# Hastings-Powell ecology

`run.py` is the execution entry point and `pipeline.py` implements generation and projection. `holling.py`, `dynamics.py`, and `fit.py` define the ecological model, integration, and parameter fitting. `plot.py`, `plot_transients.py`, and `plot_diagnostics.py` render the main experiment; `validate_multistart.py` evaluates generated fields. `results/` holds recorded main and validation reports.

See [RUNNING.md](../../docs/RUNNING.md) for commands. Write new artifacts under
`outputs/hp/`; `results/` directories contain recorded reference metrics.
