import numpy as np

CONFIGS = [
    {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
]

def get_scale_bias(mean, std):
    scale = [1.0 / float(s) for s in std]
    bias = [-float(m) / float(s) for m, s in zip(mean, scale)]
    return scale, bias

def verify_drift(ref_out, got_out, threshold):
    num = np.linalg.norm(ref_out - got_out)
    den = np.linalg.norm(ref_out) + 1e-12
    rel = float(num / den)
    return rel <= threshold

def size_ratio(fp32_bytes, fp16_bytes):
    return float(fp32_bytes) / float(fp16_bytes)
