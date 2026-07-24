import numpy as np

def effective_bits_per_weight(b, group_size):
    b_arr = np.asarray(b, dtype=np.float64)
    g_arr = np.asarray(group_size, dtype=np.float64)
    return b_arr + 32.0 / g_arr
