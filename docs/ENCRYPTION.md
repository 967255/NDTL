# Chaotic image encryption

`experiments/encryption/run.py` evaluates parent and fused systems with the same
permutation-diffusion pipeline. The article selects candidate index 6 from the
nine-point Lorenz-content/Chen-style scan and uses camera, moon, and astronaut
images. Recorded encryption metrics are in `experiments/encryption/results/`.

```bash
python -m experiments run style-scan --output outputs/encryption/fields
python -m experiments run crypto-smoke --output outputs/encryption/smoke -- --vec-path outputs/encryption/fields/vecs.npy
python -m experiments run crypto-main --output outputs/encryption/main -- --vec-path outputs/encryption/fields/vecs.npy
```

The smoke command reduces the image count, image size, and evaluation workload.
A custom scan must use candidate indices matching that scan's recorded weights.
