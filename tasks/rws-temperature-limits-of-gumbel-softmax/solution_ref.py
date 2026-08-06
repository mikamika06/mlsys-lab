import math
import numpy as np


def gumbel_temperature_limits(logits, g, tau_small, tau_large):
    logits_arr = np.asarray(logits, dtype=np.float64)
    g_arr = np.asarray(g, dtype=np.float64)
    
    n = logits_arr.shape[0]
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        values[i] = logits_arr[i] + g_arr[i]

    max_val = values[0]
    index = 0
    for i in range(1, n):
        if values[i] > max_val:
            max_val = values[i]
            index = i

    scaled = np.empty(n, dtype=np.float64)
    for i in range(n):
        scaled[i] = values[i] / float(tau_large)

    max_scaled = scaled[0]
    for i in range(1, n):
        if scaled[i] > max_scaled:
            max_scaled = scaled[i]

    exp_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        exp_values[i] = math.exp(scaled[i] - max_scaled)

    sum_exp = 0.0
    for i in range(n):
        sum_exp += exp_values[i]

    distribution = np.empty(n, dtype=np.float64)
    for i in range(n):
        distribution[i] = exp_values[i] / sum_exp

    return index, distribution.astype(np.float64)
