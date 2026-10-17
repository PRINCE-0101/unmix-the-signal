# Experiment report

## Question

Can statistically independent sources be recovered when only linear mixtures are observed?

## Method

The experiment creates three deterministic signals with different probability distributions: a speech-like harmonic signal, a stepped musical tone, and sparse percussion. A non-orthogonal matrix mixes them into three observed channels. The implementation then:

1. centres the observations;
2. whitens them with eigenvalue decomposition;
3. applies symmetric FastICA with the `tanh` contrast function;
4. aligns outputs only for evaluation, because ICA cannot determine order, sign, or amplitude;
5. measures absolute Pearson correlation and scale-invariant SDR.

Fifteen seeded initialisations are recorded in `results/benchmark.json`. All generated inputs, mixtures, outputs, plots and metrics are reproducible—no dataset download is required.

## Interpretation

A successful run has correlations close to 1 and positive SI-SDR. The mixture matrix must be full-rank and the sources must be statistically independent. ICA may fail when sources are Gaussian, dependent, too noisy, or when there are fewer microphones than sources.

## Honest limitations

The included audio is synthetic and intentionally well-conditioned. Real rooms introduce echo, time delays and moving sources, which violate the instantaneous linear mixing assumption. This repository demonstrates the core mathematics; it is not presented as a production speech-separation model.

