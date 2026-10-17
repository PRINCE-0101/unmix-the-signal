"""A compact, from-scratch FastICA implementation using symmetric decorrelation."""
from __future__ import annotations

import numpy as np


class FastICA:
    """Separate statistically independent signals from observed mixtures.

    Parameters mirror the core knobs of common ICA implementations, while the
    algorithm stays visible and readable for learning purposes.
    """

    def __init__(self, n_components=None, max_iter=1000, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _sym_decorrelation(matrix):
        values, vectors = np.linalg.eigh(matrix @ matrix.T)
        values = np.maximum(values, 1e-12)
        return (vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T) @ matrix

    def _whiten(self, observations):
        self.mean_ = observations.mean(axis=1, keepdims=True)
        centered = observations - self.mean_
        covariance = centered @ centered.T / centered.shape[1]
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][: self.n_components_]
        values = np.maximum(values[order], 1e-12)
        vectors = vectors[:, order]
        self.whitening_ = np.diag(1.0 / np.sqrt(values)) @ vectors.T
        return self.whitening_ @ centered

    def fit_transform(self, observations):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2:
            raise ValueError("observations must have shape (channels, samples)")
        channels, samples = observations.shape
        if channels < 2 or samples < channels:
            raise ValueError("need at least two channels and more samples than channels")
        self.n_components_ = self.n_components or channels
        if not 1 <= self.n_components_ <= channels:
            raise ValueError("n_components must be between 1 and channel count")

        whitened = self._whiten(observations)
        rng = np.random.default_rng(self.random_state)
        weights = self._sym_decorrelation(
            rng.normal(size=(self.n_components_, self.n_components_))
        )

        for iteration in range(1, self.max_iter + 1):
            projected = weights @ whitened
            activation = np.tanh(projected)
            derivative = 1.0 - activation**2
            updated = activation @ whitened.T / samples
            updated -= derivative.mean(axis=1)[:, None] * weights
            updated = self._sym_decorrelation(updated)
            change = np.max(np.abs(np.abs(np.diag(updated @ weights.T)) - 1.0))
            weights = updated
            if change < self.tol:
                break
        else:
            raise RuntimeError("FastICA did not converge; increase max_iter")

        self.n_iter_ = iteration
        self.components_ = weights @ self.whitening_
        separated = weights @ whitened
        separated /= np.maximum(separated.std(axis=1, keepdims=True), 1e-12)
        return separated

    def fit(self, observations):
        self.fit_transform(observations)
        return self

    def transform(self, observations):
        observations = np.asarray(observations, dtype=float)
        return self.components_ @ (observations - self.mean_)

