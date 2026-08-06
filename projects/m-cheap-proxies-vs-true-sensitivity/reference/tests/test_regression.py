import numpy as np
from sens.metric import compute_true_sensitivity


def test_true_sensitivity_activation_dependence():
    w = np.full((16, 16), 0.3, dtype=np.float32)

    layer_zero_act = {
        "layer_id": 0,
        "weight": w,
        "activation": np.zeros((8, 16), dtype=np.float32)
    }

    layer_ones_act = {
        "layer_id": 0,
        "weight": w,
        "activation": np.ones((8, 16), dtype=np.float32)
    }

    s1 = compute_true_sensitivity(layer_zero_act)
    s2 = compute_true_sensitivity(layer_ones_act)

    # If the metric is purely weight-based, s1 and s2 will be exactly equal
    # but since the metric correctly evaluates activations, they should differ.
    assert abs(s1 - s2) > 1e-5
