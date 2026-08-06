import math
from collections.abc import Sequence


def softmax_temperature_sweep(logits: list[float], temps: Sequence[float]) -> list[list[float]]:
    """Compute the temperature-scaled softmax for each temperature in `temps`."""
    n_temps = len(temps)
    n_logits = len(logits)
    probs = []

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

        row = []
        for j in range(n_logits):
            row.append(exp_shifted[j] / sum_exp)
        probs.append(row)

    return probs
