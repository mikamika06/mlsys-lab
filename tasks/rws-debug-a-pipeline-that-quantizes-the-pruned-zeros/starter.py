def compound_prune_quantize_2_4(W: list[list[float]], nbits: int=4) -> list[list[float]]:
    """Compound 2:4 structured pruning + per-group int quantization.

    W: 2-D float array, last dimension a multiple of 4. Every consecutive
    block of 4 elements along the last axis: zero the 2 smallest-magnitude
    elements (keep the 2 largest survivors), compute a per-block scale from
    the survivor magnitudes, then quantize/dequantize the survivors with
    that scale (qmax = 2 ** (nbits - 1) - 1). Pruned positions stay 0.0.

    Returns W_hat: float64, same shape as W.
    """
    raise NotImplementedError('your code here')
