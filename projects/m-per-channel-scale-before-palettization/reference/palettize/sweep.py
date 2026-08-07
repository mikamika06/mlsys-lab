import numpy as np
from palettize.decode import decode_weight
from palettize.scale import per_channel_scale


def pareto_sweep(weight: np.ndarray, nbits_list: list):
    results = []
    for nbits in nbits_list:
        q, s, z = per_channel_scale(weight, nbits)
        dec = decode_weight(q, s, z)
        mse = float(np.mean((weight - dec) ** 2))
        size_bytes = (weight.size * nbits) / 8.0
        results.append({"nbits": nbits, "mse": mse, "size_bytes": size_bytes})
    return results
