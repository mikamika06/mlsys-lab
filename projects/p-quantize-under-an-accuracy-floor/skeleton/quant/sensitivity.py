def compute_layer_sensitivity(model, evaluator, calibration_data):
    """Measures accuracy degradation when quantizing each layer individually."""
    raise NotImplementedError


def select_mixed_precision_config(sensitivity_scores, target_bits=5.5):
    """Assigns bit-precision per layer based on sensitivity scores."""
    raise NotImplementedError
