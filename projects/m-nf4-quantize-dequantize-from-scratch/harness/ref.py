import numpy as np


def norm_ppf(p):
    t = np.sqrt(-2.0 * np.log(p if p < 0.5 else 1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    val = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    return -val if p < 0.5 else val


def build_int4_codebook():
    return np.linspace(-1.0, 1.0, 16)


def build_fp4_codebook():
    return np.array([
        -1.0, -0.666, -0.5, -0.333, -0.25, -0.166, -0.083, -0.0,
         0.0,  0.083,  0.166,  0.25,  0.333,  0.5,  0.666,  1.0
    ])


def build_nf4_codebook():
    p_neg = np.linspace(0.03, 0.5, 8)
    d = p_neg[1] - p_neg[0]
    p_pos = np.linspace(0.5 + d, 0.97, 8)
    p_all = np.concatenate([p_neg, p_pos])
    quantiles = np.array([norm_ppf(p) for p in p_all])
    return quantiles / np.max(np.abs(quantiles))


def quantize_blockwise(tensor, codebook, block_size=64):
    N = len(tensor)
    num_blocks = N // block_size
    tensor = tensor.reshape(num_blocks, block_size)

    absmax = np.max(np.abs(tensor), axis=1)
    scale = np.where(absmax == 0, 1.0, absmax)

    normalized = tensor / scale[:, None]

    diffs = np.abs(normalized[..., None] - codebook)
    indices = np.argmin(diffs, axis=-1).astype(np.uint8)
    indices = indices.reshape(-1)

    high = indices[0::2] << 4
    low = indices[1::2]
    quantized = high | low

    return quantized, absmax


def dequantize_blockwise(quantized, absmax, codebook, block_size=64):
    high = (quantized >> 4) & 0x0F
    low = quantized & 0x0F

    indices = np.empty(len(quantized) * 2, dtype=np.uint8)
    indices[0::2] = high
    indices[1::2] = low

    values = codebook[indices]
    values = values.reshape(-1, block_size)
    values = values * absmax[:, None]

    return values.reshape(-1)
