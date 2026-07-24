import numpy as np


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    i = 0
    for i in range(max_exponent_bits):
        if signed:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
        else:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += list((10.0 ** (-(max_exponent_bits - 1) + i)) * means)
        if signed:
            data += list(-(10.0 ** (-(max_exponent_bits - 1) + i)) * means)
    if additional_items > 0:
        boundaries = np.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += list((10.0 ** (-(max_exponent_bits - 1) + i)) * means)
        if signed:
            data += list(-(10.0 ** (-(max_exponent_bits - 1) + i)) * means)
    data.append(0.0)
    data.append(1.0)
    data.sort()
    return np.array(data, dtype=np.float64)
