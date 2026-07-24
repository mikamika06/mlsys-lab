import numpy as np


def map_mask_to_arch(layer_gates, head_gates, dim_gates, target):
    target_L, target_H, target_d_ff = target

    def top_indices(values, count):
        values = np.asarray(values)
        order = np.lexsort((np.arange(values.shape[0]), -values))
        return np.sort(order[:count]).tolist()

    layers = top_indices(layer_gates, target_L)

    heads = []
    for layer in layers:
        heads.append(top_indices(head_gates[layer], target_H))

    dims = top_indices(dim_gates, target_d_ff)

    return {
        "layers": layers,
        "heads": heads,
        "dims": dims,
    }
