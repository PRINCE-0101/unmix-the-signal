# Five-minute presentation guide

## 1. Hook (30 seconds)

Play one mixed channel, then a recovered channel. Ask: “Can a computer separate sounds without ever being told what each sound is?”

## 2. Problem (45 seconds)

Explain the cocktail-party problem. Microphones record weighted sums: **X = A·S**. We know X, but both the source matrix S and mixing matrix A are hidden.

## 3. Idea (90 seconds)

- Centre and whiten the recordings so their scales and correlations do not dominate.
- Search for rotations whose outputs are maximally non-Gaussian.
- Statistical independence is stronger than ordinary uncorrelatedness.
- FastICA uses a fixed-point update, so there is no labelled training dataset.

## 4. Live demonstration (90 seconds)

Run `python run.py`, open `results/demo/index.html`, compare original and recovered audio, then show `results/separation_overview.png` and `results/metrics.json`.

## 5. Results and honesty (45 seconds)

Mention the permutation/sign ambiguity and why evaluation aligns outputs. End with limitations: synthetic instantaneous mixtures are easier than reverberant real rooms.

