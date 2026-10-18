# Unmix the Signal

A reproducible, beginner-friendly **blind source separation** project using PCA whitening and a from-scratch implementation of the FastICA algorithm.

## Live demo

**https://unmix-the-signal.onrender.com**

The live showcase lets you:

- listen to each original source and its recovered FastICA output;
- compare recovery quality using correlation and SI-SDR;
- inspect original, mixed and separated waveforms;
- follow the full source-generation → mixing → whitening → FastICA → evaluation pipeline.

## What the project does

The program generates three independent synthetic sources — speech-like, music-like and percussion-like — and mixes them as if three microphones heard different combinations. FastICA receives only the observed mixtures and estimates the hidden independent sources.

The project intentionally implements the key separation logic for learning rather than simply wrapping scikit-learn.

## Current generated result

The committed demo run uses:

- **3 hidden sources** and **3 observed mixtures**;
- **8 kHz** sample rate and **24,000 samples**;
- **5 FastICA iterations** in the generated result;
- mean absolute correlation of approximately **0.999991**;
- mean SI-SDR of approximately **49.70 dB**.

These values are read from `results/metrics.json` by the deployed showcase.

## Repository structure

```text
unmix-the-signal/
├── unmix/                  # FastICA, audio helpers, metrics and pipeline
├── tests/                  # recovery and validation tests
├── results/
│   ├── audio/              # original, mixture and separated WAV files
│   ├── demo/index.html     # polished interactive listening showcase
│   ├── metrics.json        # generated evaluation metrics
│   └── separation_overview.png
├── REPORT.md               # method, interpretation and limitations
├── PRESENTATION_GUIDE.md   # five-minute demo flow
├── render.yaml             # Render static-site deployment
├── requirements.txt
└── run.py
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Then open:

```text
results/demo/index.html
```

To repeat the separation with 15 different initialisations:

```bash
python run.py --benchmark
```

Run the automated tests with:

```bash
pytest -q
```

## Mathematical model

The observed microphone signals follow

**X = A·S**

where:

- **S** contains the unknown independent sources;
- **A** is the unknown mixing matrix;
- **X** contains only the observed mixtures.

The pipeline centres and PCA-whitens **X**, then searches for an unmixing matrix **W** such that

**Y = W·X**

contains components that are as statistically independent and non-Gaussian as possible. The FastICA fixed-point update uses `tanh` as the contrast function and symmetric decorrelation to estimate all components together.

ICA can recover source order, sign and amplitude only up to ambiguity. For evaluation, the project finds the best permutation and sign before reporting absolute correlation and scale-invariant signal-to-distortion ratio (SI-SDR).

## Outputs

A normal run generates:

- 3 original WAV files;
- 3 mixed WAV files;
- 3 separated WAV files;
- `results/separation_overview.png`;
- `results/metrics.json`;
- the interactive listening page in `results/demo/`.

Benchmark mode additionally creates `results/benchmark.json`.

## Showcase features

The deployed page includes:

- responsive project hero section;
- live metric cards loaded from the generated result;
- original-vs-recovered audio comparison;
- before-vs-after explanation;
- four-step algorithm walkthrough;
- waveform overview;
- technology and project-highlight sections;
- direct links to the repository and report.

## Scope and limitations

This is an educational, controlled experiment designed to make ICA assumptions measurable and easy to demonstrate. Real-room audio separation is significantly harder because it may involve reverberation, time delays, additive noise, moving sources, non-linear effects and a changing acoustic environment.

## Deployment

The project is deployed as a Render static site using `render.yaml`. The generated files under `results/` are served directly, and the root URL rewrites to the listening demo.

## License

MIT
