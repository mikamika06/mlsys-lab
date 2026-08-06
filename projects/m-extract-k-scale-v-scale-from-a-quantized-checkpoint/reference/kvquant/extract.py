import numpy as np

def extract_scales(checkpoint):
    scales = {}
    for k, v in checkpoint.items():
        if "k_scale" in k or "v_scale" in k:
            scales[k] = np.array(v, dtype=np.float32)
    return scales
