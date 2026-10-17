# Unmix the Signal

A reproducible, beginner-friendly demonstration of **blind source separation** using PCA whitening and a from-scratch implementation of the FastICA algorithm.

The program generates three source signals, mixes them as if three microphones heard different combinations, and then recovers the sources without seeing the originals during separation. It also produces playable WAV files, a waveform plot, correlation/SI-SDR metrics, and a small offline listening page.

## What is inside

```text
unmix-the-signal/
├── unmix/                  # FastICA, audio helpers, metrics and pipeline
├── tests/                  # recovery and validation tests
├── results/demo/index.html # offline listening demo
├── REPORT.md               # method, interpretation and limitations
├── PRESENTATION_GUIDE.md   # five-minute demo flow
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

Open `results/demo/index.html` after the run. To repeat the separation with 15 different initialisations:

```bash
python run.py --benchmark
```

Run the tests with:

```bash
pytest -q
```

## Mathematical model

The microphones observe **X = A·S**, where the rows of S are unknown independent sources and A is an unknown mixing matrix. The project centres and whitens X, then searches for an unmixing matrix W whose outputs **Y = W·X** are maximally non-Gaussian. FastICA uses a fixed-point update with `tanh` as the contrast function and symmetric decorrelation to estimate all components together.

ICA can recover order, sign and amplitude only up to ambiguity. The evaluation therefore finds the best permutation and sign before reporting absolute correlation and scale-invariant signal-to-distortion ratio (SI-SDR).

## Expected generated files

- nine WAV files: three originals, mixtures and separated outputs;
- `results/separation_overview.png`;
- `results/metrics.json`;
- `results/benchmark.json` after benchmark mode.

## Scope

This is an educational implementation rather than a wrapper around scikit-learn. The synthetic experiment makes the assumptions measurable and repeatable. Real-room audio would additionally require handling reverberation, delays, noise and source movement.

## License

MIT

