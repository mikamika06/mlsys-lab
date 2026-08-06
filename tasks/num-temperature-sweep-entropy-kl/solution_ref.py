import math
import numpy as np


def softmax_temperature_sweep(logits: np.ndarray, temps):
    """Compute the temperature-scaled softmax for each temperature in `temps`."""
    logits = np.asarray(logits, dtype=np.float64)
    temps = np.asarray(temps, dtype=np.float64)

    n_temps = len(temps)
    n_logits = len(logits)
    probs = np.zeros((n_temps, n_logits), dtype=np.float64)

    for i in range(n_temps):
        t = temps[i]

        scaled = []
        for j in range(n_logits):
            scaled.append(logits[j] / t)

        max_val = scaled[0]
        for j in range(1, n_logits):
            if scaled[j] > max_val:
                max_val = scaled[j]

        exp_shifted = []
        sum_exp = 0.0
        for j in range(n_logits):
            val = math.exp(scaled[j] - max_val)
            exp_shifted.append(val)
            sum_exp += val

        for j in range(n_logits):
            probs[i, j] = exp_shifted[j] / sum_exp

    return probs
