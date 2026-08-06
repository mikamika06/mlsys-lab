import numpy as np


def extract_scales(state_dict):
    k_scales = {}
    v_scales = {}
    for k, v in state_dict.items():
        if "k_scale" in k:
            layer_idx = int(k.split(".")[1])
            k_scales[layer_idx] = np.array(v, dtype=np.float32)
        elif "v_scale" in k:
            layer_idx = int(k.split(".")[1])
            v_scales[layer_idx] = np.array(v, dtype=np.float32)
    return k_scales, v_scales
