import numpy as np

NF4_CODEBOOK = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250929000305176,
    -0.39491748809814453,
    -0.28444117307662964,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791837096214294,
    0.4407098293384552,
    0.5626170039176941,
    0.722956836227417,
    1.0,
], dtype=np.float64)


def quantize_nf4(x, block_size=64):
    orig_shape = x.shape
    flat = x.astype(np.float64).flatten()
    n = len(flat)
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        padded = np.pad(flat, (0, pad_len), mode="constant", constant_values=0)
    else:
        padded = flat
    blocks = padded.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    norm_blocks = blocks / scales
    diffs = np.abs(norm_blocks[:, :, None] - NF4_CODEBOOK[None, None, :])
    q_indices = np.argmin(diffs, axis=2).astype(np.uint8)
    return q_indices, scales.squeeze(axis=1), orig_shape, n


def dequantize_nf4(q_indices, scales, orig_shape, orig_len):
    scales_arr = np.atleast_1d(scales)
    dequant_blocks = NF4_CODEBOOK[q_indices] * scales_arr[:, None]
    flat = dequant_blocks.flatten()[:orig_len]
    return flat.reshape(orig_shape)


def measure_nf4_cycle_error(w_init, num_cycles, update_fn=None, block_size=64):
    w_ref = w_init.astype(np.float64).copy()
    w_q = w_init.astype(np.float64).copy()
    errors = []
    for c in range(num_cycles):
        q_idx, scales, shape, orig_len = quantize_nf4(w_q, block_size=block_size)
        w_dequant = dequantize_nf4(q_idx, scales, shape, orig_len)
        if update_fn is not None:
            delta_q = update_fn(c, w_dequant)
            delta_ref = update_fn(c, w_ref)
            w_q = w_dequant + delta_q
            w_ref = w_ref + delta_ref
        else:
            w_q = w_dequant
        norm_ref = np.linalg.norm(w_ref)
        err = float(np.linalg.norm(w_q - w_ref) / (norm_ref + 1e-12))
        errors.append(err)
    return errors
