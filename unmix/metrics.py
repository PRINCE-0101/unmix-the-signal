"""Permutation-aware evaluation for the scale/sign ambiguity of ICA."""
from __future__ import annotations

from itertools import permutations
import numpy as np


def correlation_matrix(reference, estimate):
    reference = reference - reference.mean(axis=1, keepdims=True)
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    reference /= np.maximum(reference.std(axis=1, keepdims=True), 1e-12)
    estimate /= np.maximum(estimate.std(axis=1, keepdims=True), 1e-12)
    return reference @ estimate.T / reference.shape[1]


def best_alignment(reference, estimate):
    correlations = correlation_matrix(reference, estimate)
    best = max(permutations(range(reference.shape[0])), key=lambda p: sum(abs(correlations[i, p[i]]) for i in range(len(p))))
    aligned = estimate[list(best)].copy()
    for index in range(reference.shape[0]):
        aligned[index] *= np.sign(correlations[index, best[index]]) or 1.0
        aligned[index] *= np.dot(reference[index], aligned[index]) / max(np.dot(aligned[index], aligned[index]), 1e-12)
    return aligned, list(best), correlations


def si_sdr(reference, estimate):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    target = np.dot(estimate, reference) * reference / max(np.dot(reference, reference), 1e-12)
    noise = estimate - target
    return 10 * np.log10(max(np.dot(target, target), 1e-12) / max(np.dot(noise, noise), 1e-12))

