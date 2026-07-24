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
    blocks = W.reshape(*shape[:-1], -1, 4)
    abs_blocks = np.abs(blocks)

    # Indices of the two largest-magnitude elements per block.
    order = np.argsort(abs_blocks, axis=-1)
    keep_mask = np.zeros_like(blocks, dtype=bool)
    np.put_along_axis(keep_mask, order[..., 2:], True, axis=-1)

    pruned = np.where(keep_mask, blocks, 0.0)
    survivor_abs = np.where(keep_mask, abs_blocks, 0.0)

    # Scale from the survivors ONLY: sum of survivor magnitudes divided by
    # the number of actual survivors in that block (normally 2).
    count = keep_mask.sum(axis=-1, keepdims=True).astype(np.float64)
    sum_abs = survivor_abs.sum(axis=-1, keepdims=True)
    scale = np.where(count > 0, sum_abs / np.maximum(count, 1.0), 1.0)

    qmax = 2 ** (nbits - 1) - 1
    code = np.clip(np.round(pruned / scale), -qmax, qmax)
    dequant = np.where(keep_mask, code * scale, 0.0)

    return dequant.reshape(shape)
