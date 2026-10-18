# Included CAE checkpoints

Only these seven publication checkpoints are included. They are copied from the
recorded experiments without retraining or changing their tensor values.

| File under `checkpoints/` | Role | CAE depth / grid |
|---|---|---|
| `hp/cae.pt` | HP `fixed02/main64/stable_to_chaotic` main experiment | 96 / 64 cubed |
| `lorenz_like/lorenz_chen_lv.pt` | Normalized Lorenz, Chen, and Lv fusion and content/style scan | 64 / 48 cubed |
| `lorenz_like/lorenz_chen_lv_original.pt` | Original-coordinate Lorenz feature perturbation | 64 / 48 cubed |
| `lorenz_like/lorenz_lv_rossler.pt` | Normalized Lorenz, Lv, and Rossler experiments | 64 / 48 cubed |
| `lorenz_like/lorenz_hadley_halvorsen.pt` | Normalized Lorenz, Hadley, and Halvorsen experiments | 64 / 48 cubed |
| `lorenz_like/lorenz_nose_hoover_rucklidge.pt` | Normalized Lorenz, Nose-Hoover, and Rucklidge experiments | 64 / 48 cubed |
| `seirs/cae.pt` | SEIRS coarse/fine transfer experiments | 64 / 48 cubed |

The shared `ndtl.models.CAE` preserves the original encoder/decoder parameter
names. HP uses its depth-96, stride-(2,2,2) configuration through `EcologyCAE`.
The HP file also stores its profile, ecological domain, and parent parameters.

Per-pair classification models, HP smoke/alternative-profile models, intermediate
training checkpoints, and historical model versions are excluded. Newly trained
models are written under `outputs/` and are ignored by Git.
