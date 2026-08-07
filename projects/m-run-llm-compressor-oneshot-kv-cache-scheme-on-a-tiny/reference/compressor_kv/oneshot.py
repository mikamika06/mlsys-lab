import numpy as np

def compute_kv_scales(config, activations):
    scales = []
    for act in activations:
        mx = float(np.max(np.abs(act)))
        scale = mx / 448.0 if mx > 0 else 1.0
        scales.append(scale)
    mean_scale = float(np.mean(scales)) if scales else 1.0
    return {"scale": mean_scale, "scheme": "fp8_e4m3"}
