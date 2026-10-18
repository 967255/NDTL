# Publication experiment index

The machine-readable index is [experiments/catalog.json](../experiments/catalog.json).
All executable experiments live under `experiments/`; shared numerical code is in `ndtl/`.

| ID | Experiment | Paper location | Implementation |
|---|---|---|---|
| E01 | Lorenz-Lv content fusion | Fig. 2, PDF p. 4 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E02 | Lorenz-Chen content fusion | Fig. 11, PDF p. 14 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E03 | Lorenz-Halvorsen content fusion | Fig. 12, PDF p. 14 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E04 | Lorenz-Nose-Hoover content fusion | Fig. 13, PDF p. 14 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E05 | Lv-Rossler content fusion | Fig. 14, PDF p. 14 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E06 | Hadley-Halvorsen content fusion | Fig. 15, PDF p. 15 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E07 | Nose-Hoover-Rucklidge content fusion | Fig. 16, PDF p. 16 | `experiments/classic/run.py`, `experiments/classic/evaluate.py`, `experiments/classic/plot.py` |
| E08 | Lorenz content and Chen style weight scan | Table I, PDF p. 4; Sec. III D | `experiments/classic/run.py`, `experiments/classic/evaluate.py` |
| E09 | Content and Gram feature transformations | Table II, Appendix  B, PDF p. 12 | `experiments/perturbation/transformations.py`, `ndtl/fields.py` |
| E10 | Layer perturbation and feature interpretation | Fig. 3, Sec. III E, PDF p. 5 | `experiments/perturbation/run.py`, `experiments/perturbation/analyze.py` |
| E11 | Representative Lyapunov spectrum landscapes | Fig. 4, PDF p. 5 | `experiments/classification/plot_spectra.py` |
| E12 | 52-system classification | Fig. 5, Sec. IV B, PDF p. 7 | `experiments/classification/train52.py`, `experiments/classification/train20.py`, `experiments/classification/plot_landscape.py` |
| E13 | Hierarchical and spectral clustering | Fig. 6, Sec. IV C, PDF p. 7 | `experiments/classification/plot_strips.py`, `experiments/classification/plot_hierarchy.py` |
| E14 | 52-system reordered distance matrix | Fig. 18, Appendix  E, PDF p. 17 | `experiments/classification/plot_spectral.py` |
| E15 | 20-system classification | Fig. 19, Appendix  E, PDF p. 17 | `experiments/classification/train20.py`, `experiments/classification/plot_spectral.py` |
| E16 | 3D similarity landscape | Fig. 20, Appendix  E, PDF p. 17 | `experiments/classification/plot_landscape.py` |
| E17 | Hastings-Powell ecology | Fig. 7; Table III, IV; Sec. V A and Appendix  F; PDF p. 8, 18 | `experiments/hp/run.py`, `experiments/hp/pipeline.py`, `experiments/hp/fit.py`, `experiments/hp/plot.py` |
| E18 | Raw and projected HP multistart validation | Table V, Appendix  F d, PDF p. 19 | `experiments/hp/validate_multistart.py` |
| E19 | Saved HP periodic-orbit validation | Appendix  F e, PDF p. 19 | Reference record |
| E20 | SEIRS interpretable models | Fig. 8; Sec. V B and Appendix  C; PDF p. 8, 9, 15 | `experiments/seirs/run.py`, `experiments/seirs/analyze.py`, `experiments/seirs/polynomial.py`, `experiments/seirs/plot.py` |
| E21 | Chaotic image encryption | Fig. 9, Table VI; Sec. V C and Appendix  G; PDF p. 9, 19 | `experiments/encryption/run.py`, `experiments/classic/run.py` |
| E23 | NDTL framework schematic | Fig. 1, PDF p. 2 | Reference record |
| E25 | Hamiltonian feature interpretation | Sec. III B, PDF p. 3 | Reference record |

Training and generation commands are in [RUNNING.md](RUNNING.md).
Reference entries identify included records or paper context; they are not additional executable pipelines.
