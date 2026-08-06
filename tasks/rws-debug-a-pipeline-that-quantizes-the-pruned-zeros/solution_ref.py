import numpy as np


def compound_prune_quantize_2_4(W, nbits=4):
    """Compound 2:4 structured pruning + per-group int quantization.

    `W` is a 2-D float array whose last dimension is a multiple of 4. Every
    consecutive block of 4 elements along the last axis is treated as one
    N:M block *and* one quantization group:

    1. Structured 2:4 prune: zero the 2 smallest-magnitude elements of the
       block, keep the 2 largest (the "survivors").
    2. The block's quantization scale is the mean magnitude of its
       survivors only:

           scale = mean(|survivors|)

       (if a block had no survivors to begin with -- i.e. it was already
       all zero -- use `scale = 1.0`).
    3. Each survivor `v` is quantized/dequantized in place:
       `code = clip(round(v / scale), -qmax, qmax)`, `dequant = code *
       scale`, with `qmax = 2 ** (nbits - 1) - 1`. Pruned positions stay
       exactly `0.0` -- they need no code at all.

    Parameters
    ----------
    W : np.ndarray, shape (..., 4k)
    nbits : int

    Returns
    -------
    W_hat : np.ndarray, float64, same shape as W
    """
    W = np.asarray(W, dtype=np.float64)
    shape = W.shape
    flat_W = W.reshape(-1, 4)
    out_flat = np.empty_like(flat_W)
    qmax = 2 ** (nbits - 1) - 1

    for i in range(flat_W.shape[0]):
        block = flat_W[i]
        abs_block = [abs(block[0]), abs(block[1]), abs(block[2]), abs(block[3])]
        indexed = [(0, abs_block[0]), (1, abs_block[1]), (2, abs_block[2]), (3, abs_block[3])]
        sorted_indexed = sorted(indexed, key=lambda x: x[1])
        survivor_indices = {sorted_indexed[2][0], sorted_indexed[3][0]}

        sum_abs = 0.0
        for j in range(4):
            if j in survivor_indices:
                sum_abs += abs_block[j]

        scale = sum_abs / 2.0

        for j in range(4):
            if j in survivor_indices:
                v = block[j]
                c = round(v / scale)
                if c > qmax:
                    c = qmax
                elif c < -qmax:
                    c = -qmax
                out_flat[i, j] = c * scale
            else:
                out_flat[i, j] = 0.0

    return out_flat.reshape(shape)
