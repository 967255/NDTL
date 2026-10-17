# NDTL research website

A static, dark research website for *Dynamics creation through neural dynamical transfer learning*, Physical Review Research **8**, 033337 (2026), DOI: [10.1103/ccgv-hgjb](https://doi.org/10.1103/ccgv-hgjb).

The website follows Motivation → Lorenz-like fusion → Dynamics classification → HP / SEIRS / image encryption → Method. The poster layout keeps explanations in expandable notes and the paper. It contains no online inference or training service.

Navigation remains expanded at every screen width. The header's **Star on GitHub** link opens `https://github.com/967255/NDTL`, where visitors can star the repository using their own GitHub account.

The fusion section displays Lorenz + Lv’s and Lorenz + Chen’s, each with nine evaluated weights in a horizontal gallery. Hovering or selecting a trajectory highlights its Lyapunov and dimension signatures. The content–style experiment is retained in the downloadable research data. Classification links real spectrum paths, four ranked components, the distance matrix, and a rotatable similarity landscape with bright cluster markers.

Landscape points toggle on click or tap. Zero, one, and two selections are supported; fewer than two selections clear the pair-specific curves and distance. Click blank space, press Escape while the landscape has focus, or use **Clear selection** to clear both points. Dragging rotates the view without changing the selection.

## Public preview and editing

Website: <https://967255.github.io/NDTL/>. The website source lives in `website/` in the existing NDTL repository. It is a preview that can continue to change at the same URL.

Edit `dist/index.html`, `dist/styles.css` and `dist/app.js` in this directory. After changing JSON data, run `python3 tools/package_site.py`. Push website changes to `main`; `.github/workflows/website-pages.yml` bundles the data and publishes `website/dist` to GitHub Pages. Research training and inference code is not executed by this workflow.

For local preview:

Open `dist/index.html` directly in a browser. The local research bundle makes the interactive figures work without a server. Alternatively:

```sh
python3 tools/serve.py
```

Then open the printed `http://127.0.0.1:8765` URL. No npm installation or external JavaScript service is required.

## Publishing configuration

The repository's Settings → Pages uses **GitHub Actions** as its build source. The workflow runs after changes to `website/` or `.github/workflows/website-pages.yml`, and can also be run manually from the Actions tab. Its `github-pages` environment records the published URL. A copy of the workflow is retained in `integration/github-pages.yml`.

## Structure

```text
website/
├── dist/
│   ├── index.html, styles.css, app.js
│   ├── data/                    # audited JSON and the browser bundle
│   └── assets/                  # motivation artwork, paper, and figures
├── tools/
│   ├── export_research_data.py  # fusion and classification data
│   ├── export_classification_landscape.py # all 1,326 paths and the similarity surface
│   ├── prepare_applications.py  # application and feature-analysis figures
│   ├── package_site.py          # browser data bundle and asset checksums
│   └── serve.py                 # optional local preview
└── integration/github-pages.yml
```

Python's standard library is sufficient for bundling. No local preview archive is created by default. The source-data preparation scripts have their own scientific dependencies and source paths; they are not run during Pages publication.

## Scientific provenance

- The content–content galleries use nine real weights from the pair-specific 52-system classification benchmark (Sec. IV), re-integrated from its saved full-precision polynomial coefficients. The original Fig. 2/11 five-weight experiment is retained separately as `referenceFrames` metadata; the experiments are not mixed. The displayed weight is the Lorenz share, so it is the reverse of the benchmark's archived alpha.
- Content–content Kaplan–Yorke dimensions are calculated from the archived spectrum. Intrinsic dimension and JS divergences were not recorded for this benchmark and remain null.
- Content–style signatures reproduce Table I. Its display trajectories are re-integrated from a full-precision polynomial projection of the original saved vector-field tensors. Rounded log equations are used only as a consistency check.
- The style-only endpoint is flagged as divergent in the archive. This case is retained in the downloadable data and is not included in the displayed fusion gallery.
- Classification includes all 1,326 pair paths (11,934 evaluations), their four raw components and percentile ranks, and the original 52 × 52 matrix. Failed evaluations remain gaps. For this view alpha is the selected system B's share.
- The landscape uses Fig. 5's archived spectral coordinates and cluster memberships. The height is U = −log(mean RBF affinity to the other 51 systems), not physical energy. Its 51 × 51 display surface uses the original plotting script's IDW interpolation. Selecting a pair highlights systems without rebuilding the global surface. The embedding does not determine the clustering labels.
- The translucent terrain tint encodes U, while point colors encode clusters. Contours are calculated from the same 51 × 51 display grid with marching squares; the scientific heights, distances and clustering are unchanged.
- HP plots are integrations of saved parent and projected HP-family model parameters. The ecological conclusion concerns the projected model, not the raw optimized tensor field.
- The SEIRS view reproduces the published figure because the full numerical sweeps are not in the compact source preview. Its β/γ is not represented as the general R₀ of the new equations.
- The input-layer feature demonstration computes an actual sampled Lorenz vector field and its Gram matrix. It demonstrates exact time-reversal invariance at the input layer, not general invariance of every learned feature.
- The perturbation chart uses 390 archived records, with means and interquartile ranges across 10 seeds for each layer/noise setting.
- Full source paths, checksums and qualifications remain in the downloadable JSON. `dist/site-manifest.json` records website asset checksums.

## Image and paper credits

The triptych shows Leonardo da Vinci's Mona Lisa (content), Juan Gris's Fantômas (1915; style, CC0 via National Gallery of Art), and a new illustration made using both references with the built-in ImageGen tool. It is an illustrative analogy, not an experimental result of Gatys neural style transfer or NDTL. The exact prompt is in `dist/assets/motivation/fusion-illustration-prompt.txt`; credits and official sources are in `dist/assets/motivation/sources.json`.

The three artworks use equal display frames with proportion-preserving CSS crops. Original image files are retained. The header spells out Neural Dynamical Transfer Learning, with N, D, T and L in red.

Gatys et al. (2015) used the Tübingen Neckarfront cityscape as the content image in the original paper. The Mona Lisa example is the website's conceptual illustration.

The NDTL paper and its original figures are attributed to He Ma, Qiyang Ge, Yu Meng, Celso Grebogi, and Wei Lin, PRR **8**, 033337 (2026), DOI 10.1103/ccgv-hgjb, CC BY 4.0. The downloadable paper and original figures retain their publication information.

The website shares the existing NDTL repository; no separate website repository is required.
