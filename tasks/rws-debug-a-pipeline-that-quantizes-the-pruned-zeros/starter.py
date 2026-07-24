import numpy as np


def compound_prune_quantize_2_4(W, nbits=4):
    """Compound 2:4 structured pruning + per-group int quantization.

    W: 2-D float array, last dimension a multiple of 4. Every consecutive
    block of 4 elements along the last axis: zero the 2 smallest-magnitude
    elements (keep the 2 largest survivors), compute a per-block scale from
    the survivor magnitudes, then quantize/dequantize the survivors with
    that scale (qmax = 2 ** (nbits - 1) - 1). Pruned positions stay 0.0.

    Returns W_hat: float64, same shape as W.
    """
    W = np.asarray(W, dtype=np.float64)
    shape = W.shape
    blocks = W.reshape(*shape[:-1], -1, 4)
    abs_blocks = np.abs(blocks)

    order = np.argsort(abs_blocks, axis=-1)
    keep_mask = np.zeros_like(blocks, dtype=bool)
    np.put_along_axis(keep_mask, order[..., 2:], True, axis=-1)

    pruned = np.where(keep_mask, blocks, 0.0)

    # BUG: averages the survivor magnitudes over ALL 4 slots in the block
    # (including the two structurally-zeroed positions) instead of over
    # just the surviving elements. The pruned zeros dilute the group's
    # scale statistic and silently halve every block's scale.
    sum_abs = np.sum(np.abs(pruned), axis=-1, keepdims=True)
    scale = sum_abs / 4.0
    scale = np.where(scale == 0, 1.0, scale)

    qmax = 2 ** (nbits - 1) - 1
    code = np.clip(np.round(pruned / scale), -qmax, qmax)
    dequant = np.where(keep_mask, code * scale, 0.0)

    return dequant.reshape(shape)
