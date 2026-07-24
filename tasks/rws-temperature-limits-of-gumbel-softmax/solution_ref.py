import numpy as np


def gumbel_temperature_limits(logits, g, tau_small, tau_large):
    values = np.asarray(logits, dtype=np.float64) + np.asarray(g, dtype=np.float64)

    index = int(np.argmax(values))

    scaled = values / float(tau_large)
    scaled = scaled - np.max(scaled)
    exp_values = np.exp(scaled)
    distribution = exp_values / np.sum(exp_values)

    return index, distribution.astype(np.float64)
