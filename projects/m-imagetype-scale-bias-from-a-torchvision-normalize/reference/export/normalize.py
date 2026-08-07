import numpy as np


def normalize_to_scale_bias(mean, std):
    mean_arr = np.array(mean, dtype=np.float64)
    std_arr = np.array(std, dtype=np.float64)
    scale = 1.0 / (255.0 * std_arr)
    bias = -mean_arr / std_arr
    return scale.tolist(), bias.tolist()
