import numpy as np


def analyze_amplification(layer_refs, layer_tests, eps=1e-12):
    """Analyzes layer-wise relative error norms and error amplification growth."""
    layer_errors = []
    for r, t in zip(layer_refs, layer_tests):
        r_arr = np.asarray(r, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        norm_ref = np.linalg.norm(r_arr)
        diff_norm = np.linalg.norm(r_arr - t_arr)
        err = float(diff_norm / (norm_ref + eps))
        layer_errors.append(err)

    amplifications = [1.0]
    for i in range(1, len(layer_errors)):
        prev_err = layer_errors[i - 1]
        curr_err = layer_errors[i]
        amp = float(curr_err / (prev_err + eps))
        amplifications.append(amp)

    max_idx = int(np.argmax(amplifications))
    return {
        "layer_errors": layer_errors,
        "amplifications": amplifications,
        "max_amplifying_layer": max_idx,
    }
