import numpy as np
from unmix.audio import make_demo_sources, mix_sources
from unmix.fastica import FastICA
from unmix.metrics import best_alignment, correlation_matrix


def test_fastica_recovers_synthetic_sources():
    sources = make_demo_sources(sample_rate=2000, duration=1.5)
    mixtures, _ = mix_sources(sources)
    recovered = FastICA(random_state=7).fit_transform(mixtures)
    aligned, _, _ = best_alignment(sources, recovered)
    diagonal = np.abs(np.diag(correlation_matrix(sources, aligned)))
    assert diagonal.min() > 0.90


def test_rejects_one_dimensional_input():
    try:
        FastICA().fit_transform(np.arange(20))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("expected ValueError")

