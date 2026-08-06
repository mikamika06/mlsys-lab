import numpy as np
from spec.decay import compute_acceptance_metrics


def test_positional_decay():
    traces = [[True, True, False], [True, False], [True, True, True, False]]
    flat_rate, decay_curve = compute_acceptance_metrics(traces)

    assert isinstance(flat_rate, float)
    assert 0.0 <= flat_rate <= 1.0
    assert len(decay_curve) == 4
    assert decay_curve[0] == 1.0
    assert decay_curve[1] == 2.0 / 3.0
    assert abs(flat_rate - (6.0 / 9.0)) < 1e-6
