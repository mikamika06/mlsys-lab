import numpy as np
from polyiso.divergence import find_first_divergent_layer
from polyiso.stats import compute_polygraphy_stats


def test_divergence_detection_thresholds():
    ref_data = {
        "layer_1": (np.array([1.0, 2.0]), np.array([1.0, 2.000001])),
        "layer_2": (np.array([1.0, 2.0]), np.array([1.0, 2.05])),
        "layer_3": (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
    }
    first_div = find_first_divergent_layer(ref_data, rtol=1e-3, atol=1e-4)
    assert first_div == "layer_2", f"Expected layer_2, got {first_div}"


def test_error_stats_non_negative():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 1.9, 3.05])
    stats = compute_polygraphy_stats(a, b)
    assert stats["mae"] >= 0.0
    assert stats["max_abs_diff"] >= 0.0
    assert stats["rel_error"] >= 0.0
