# Five-minute presentation guide

Use the live showcase: **https://unmix-the-signal.onrender.com**

## 1. Hook — 30 seconds

Open the **Interactive listening demo** and play an original source followed by its recovered version.

Say:

> “Our system receives only mixtures of multiple hidden signals. Can it recover the individual sources without being told the originals during separation?”

This introduces the blind source separation problem immediately with something the audience can hear.

## 2. Problem — 45 seconds

Explain the cocktail-party idea: multiple microphones hear different combinations of the same underlying sources.

Use the model:

**X = A·S**

- **S** = unknown independent sources;
- **A** = unknown mixing matrix;
- **X** = observed mixtures available to the algorithm.

The separator knows **X**, but does not know **A** or **S**.

## 3. Core idea — 75 seconds

Scroll to **How the pipeline works** and explain the four cards:

1. Generate independent synthetic sources.
2. Mix them into observed channels.
3. Centre and PCA-whiten the mixtures.
4. Apply FastICA and evaluate the recovered components.

Key point: FastICA does **not** need labelled training data. It looks for components that are statistically independent and strongly non-Gaussian.

The implementation uses a fixed-point FastICA update with `tanh` as the contrast function and symmetric decorrelation.

## 4. Live demonstration — 90 seconds

### Audio

Play all three original/recovered pairs:

- speech-like;
- music-like;
- percussion-like.

### Metrics

Scroll to **Recovery quality**.

The committed run currently reports approximately:

- mean absolute correlation: **0.999991**;
- mean SI-SDR: **49.70 dB**;
- FastICA iterations: **5**.

Explain that correlation close to `1` means the recovered waveform closely follows the original, while a high SI-SDR indicates low reconstruction distortion.

### Waveforms

Scroll to **Waveform overview** and point out the three stages:

**Original sources → observed mixtures → FastICA output**

This visually shows that the middle signals are blended while the final outputs recover the independent structure.

## 5. Technical honesty — 45 seconds

Mention the standard ICA ambiguities:

- recovered source order can be permuted;
- sign can flip;
- scale can differ.

Therefore the evaluation aligns the estimates with the originals before calculating metrics.

Also explain the limitation: the current experiment uses synthetic instantaneous linear mixtures. Real rooms introduce reverberation, delays, noise, moving sources and changing acoustics, so real-world separation is harder.

## 6. Closing — 15 seconds

Finish with:

> “The goal of this project was not just to call an ML library. We implemented and demonstrated the separation pipeline in a way that can be heard, visualized, measured and reproduced.”

Then show the GitHub button on the live page if the evaluator wants to inspect the code.

## Quick viva questions

**Why whitening?**  
It removes second-order correlations and normalizes the covariance structure, reducing the remaining separation problem mainly to finding an appropriate rotation.

**Why non-Gaussianity?**  
Independent mixtures tend to become more Gaussian because of the central-limit effect. ICA therefore searches for directions that are maximally non-Gaussian.

**Is this supervised learning?**  
No. FastICA performs blind/unsupervised source separation and does not train on labelled source-output pairs.

**Why can source order change?**  
The mathematical ICA model does not contain information that uniquely labels component 1, 2 or 3, so permutation ambiguity is unavoidable.

**Why can the sign change?**  
If a source is a valid independent component, its negative is equally independent, so ICA cannot determine the original sign without extra information.
