import math
import numpy as np


def double_quant_nf4(W, block_size=64):
    codebook = np.array(
        [-1.0, -0.6961928, -0.52507305, -0.3949175,
         -0.28444138, -0.18477343, -0.09105004, 0.0,
         0.0795803, 0.1609302, 0.2461123, 0.33791524,
         0.44070983, 0.562617, 0.72295684, 1.0],
        dtype=np.float64,
    )

    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    padded = int(math.ceil(n / block_size) * block_size)
    x = np.zeros(padded, dtype=np.float64)
    for i in range(n):
        x[i] = flat[i]

    num_blocks = padded // block_size
    scales = np.zeros(num_blocks, dtype=np.float64)
    for b in range(num_blocks):
        max_val = 0.0
        for j in range(block_size):
            val = x[b * block_size + j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        if max_val == 0.0:
            scales[b] = 1.0
        else:
            scales[b] = max_val

    codes = np.zeros((num_blocks, block_size), dtype=np.intp)
    for b in range(num_blocks):
        scale = scales[b]
        for j in range(block_size):
            norm_val = x[b * block_size + j] / scale
            best_c = 0
            diff0 = norm_val - codebook[0]
            if diff0 < 0.0:
                diff0 = -diff0
            min_diff = diff0
            for c in range(1, len(codebook)):
                diff = norm_val - codebook[c]
                if diff < 0.0:
                    diff = -diff
                if diff < min_diff:
                    min_diff = diff
                    best_c = c
            codes[b, j] = best_c

    max_scale = scales[0]
    for b in range(1, num_blocks):
        if scales[b] > max_scale:
            max_scale = scales[b]

    scale_codes = np.zeros(num_blocks, dtype=np.uint8)
    dequant_scales = np.zeros(num_blocks, dtype=np.float64)
    for b in range(num_blocks):
        sc = round(scales[b] / max_scale * 255.0)
        if sc < 0:
            sc = 0
        elif sc > 255:
            sc = 255
        scale_codes[b] = int(sc)
        dequant_scales[b] = float(scale_codes[b]) * max_scale / 255.0

    out = np.zeros(n, dtype=np.float64)
    idx_flat = 0
    for b in range(num_blocks):
        dq_scale = dequant_scales[b]
        for j in range(block_size):
            if idx_flat < n:
                c_idx = codes[b, j]
                out[idx_flat] = codebook[c_idx] * dq_scale
                idx_flat += 1

    bits = float(4.0 + 8.0 / block_size + 32.0 / (block_size * 256.0))
    return out.reshape(shape), bits
