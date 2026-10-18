# Hastings-Powell ecology

`experiments.hp` contains ecological vector fields, CAE training, content fusion,
Holling projection, integration, plotting, and multistart validation.
The article uses `main64`, `fixed02`, and `stable_to_chaotic`, with a Holling
projection at fusion weight 0.5.

```bash
python -m experiments run hp-smoke
python -m experiments run hp-generate --dry-run
python -m experiments run hp-main --dry-run
python -m experiments run hp-paper
```

The generation job reuses `checkpoints/hp/cae.pt`; the main training job trains
a new model. The supplied checkpoint is accepted only for its matching profile,
domain, and parent pair. Recorded publication metrics are under
`experiments/hp/results/main`.

For newly generated reports, use the plotting/validation modules' `--report`
and output options as listed by `--help`. Raw-field and Holling-projection metrics
remain distinct.
