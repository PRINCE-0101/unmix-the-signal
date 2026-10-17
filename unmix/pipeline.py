"""End-to-end experiment pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .audio import make_demo_sources, mix_sources, save_wav
from .fastica import FastICA
from .metrics import best_alignment, si_sdr


def run_experiment(output_dir="results", seed=42, save_audio=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rate = 8000
    sources = make_demo_sources(rate)
    mixtures, mixing_matrix = mix_sources(sources)
    model = FastICA(random_state=seed)
    recovered = model.fit_transform(mixtures)
    aligned, permutation, correlations = best_alignment(sources, recovered)
    scores = [float(si_sdr(sources[i], aligned[i])) for i in range(len(sources))]

    if save_audio:
        labels = ["speech_like", "music_like", "percussion"]
        for i, label in enumerate(labels):
            save_wav(output / "audio" / f"original_{label}.wav", rate, sources[i])
            save_wav(output / "audio" / f"mixed_channel_{i + 1}.wav", rate, mixtures[i])
            save_wav(output / "audio" / f"separated_{label}.wav", rate, aligned[i])

    fig, axes = plt.subplots(3, 3, figsize=(13, 7), sharex=True)
    time = np.arange(sources.shape[1]) / rate
    for row in range(3):
        axes[row, 0].plot(time, sources[row], linewidth=0.6)
        axes[row, 1].plot(time, mixtures[row], linewidth=0.6, color="#d97706")
        axes[row, 2].plot(time, aligned[row], linewidth=0.6, color="#059669")
    for axis, title in zip(axes[0], ["Original sources", "Observed mixtures", "FastICA output"]):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.tight_layout()
    plot_path = output / "separation_overview.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    report = {
        "seed": seed,
        "sample_rate": rate,
        "samples": int(sources.shape[1]),
        "iterations": model.n_iter_,
        "mixing_matrix": mixing_matrix.tolist(),
        "best_permutation": permutation,
        "absolute_correlations": np.abs(correlations[:, permutation]).diagonal().tolist(),
        "si_sdr_db": scores,
        "mean_si_sdr_db": float(np.mean(scores)),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def benchmark(output_dir="results", experiments=15):
    records = [run_experiment(output_dir, seed=seed, save_audio=(seed == 0)) for seed in range(experiments)]
    summary = {
        "experiments": experiments,
        "mean_si_sdr_db": float(np.mean([item["mean_si_sdr_db"] for item in records])),
        "minimum_si_sdr_db": float(np.min([item["mean_si_sdr_db"] for item in records])),
        "mean_iterations": float(np.mean([item["iterations"] for item in records])),
        "runs": records,
    }
    Path(output_dir, "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

