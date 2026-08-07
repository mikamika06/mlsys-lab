import numpy as np
from kvquant.cache import KVCacheTracker
from kvquant.quantize import quantize_fp8, dequantize_fp8
from kvquant.calib import calibrate_scale

def oracle_baseline_bytes(num_layers, num_heads, head_dim, batch, seq):
    t = KVCacheTracker(num_layers, num_heads, head_dim)
    return t.record(batch, seq)

def oracle_quantize_roundtrip(tensor, scale):
    q = quantize_fp8(tensor, scale)
    dq = dequantize_fp8(q, scale)
    return q, dq

def oracle_calibrate(tensors):
    return calibrate_scale(tensors)

def oracle_simulate_dialog(tensors):
    scales = [calibrate_scale([t]) for t in tensors]
    errors = []
    for t, s in zip(tensors, scales):
        q = quantize_fp8(t, s)
        dq = dequantize_fp8(q, s)
        errors.append(float(np.mean(np.abs(t - dq))))
    return float(np.mean(errors))
