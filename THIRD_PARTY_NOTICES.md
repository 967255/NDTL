# Third-party components

The NDTL source is distributed under the MIT license in `LICENSE`.

`third_party/dysts/flows.py` and `chaotic_attractors.json`
are the equation and metadata snapshot used for the classification experiment,
from [GilpinLab/dysts](https://github.com/GilpinLab/dysts). They retain the
upstream Apache License 2.0, included as `third_party/dysts/LICENSE.md`.
The snapshot is preserved rather than replaced with the latest upstream equations.

Reference: William Gilpin, *Chaos as an interpretable benchmark for forecasting
and data-driven modelling*, NeurIPS (2021), https://arxiv.org/abs/2110.05266.

The encryption demonstration uses the standard `camera`, `moon`, and `astronaut`
images provided by scikit-image, loaded through `skimage.data` at runtime.
