import numpy as np


def quantize_fp8(x, scale, ebits=4, mbits=3):
    qmax = float((1 << (mbits + ebits)) - 1)
    if ebits == 4 and mbits == 3:
        max_val = 448.0
    else:
        max_val = float((1.0 + (float((1 << mbits) - 1)) / float(1 << mbits)) * (1 << ((1 << (ebits - 1)) - 1)))
    scaled = x / scale
    clipped = np.clip(scaled, -max_val, max_val)
    quantized = np.round(clipped)
    return quantized


def dequantize_fp8(x_q, scale):
    return x_q * scale


def compute_rel_error(x, x_rec):
    num = np.linalg.norm(x - x_rec)
    den = np.linalg.norm(x) + 1e-12
    return float(num / den)
