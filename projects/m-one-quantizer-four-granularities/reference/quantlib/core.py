import numpy as np


def quantize(tensor, qparams, bits=8, symmetric=True):
    qmin = -(1 << (bits - 1)) if symmetric else 0
    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    scale, zero_point = qparams
    if symmetric:
        scaled = np.round(tensor / scale)
    else:
        scaled = np.round(tensor / scale) + zero_point
    return np.clip(scaled, qmin, qmax).astype(np.int8 if bits <= 8 else np.int16)


def dequantize(qtensor, qparams, symmetric=True):
    scale, zero_point = qparams
    if symmetric:
        return qtensor.astype(np.float32) * scale
    else:
        return (qtensor.astype(np.float32) - zero_point) * scale
