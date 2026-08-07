def quantize_dequantize_kv(K: list[list[float]], V: list[list[float]], group_size: int, bits: int=4) -> tuple[list[list[float]], list[list[float]]]:
    """Asymmetric int4 (default) group quant-dequant, KIVI-style.

    K: (seq_len, channels) float64 -- quantized PER-CHANNEL, i.e. each
        column is split into contiguous groups of `group_size` TOKENS
        (rows), each group getting its own scale/zero-point.
    V: (seq_len, channels) float64 -- quantized PER-TOKEN, i.e. each row
        is split into contiguous groups of `group_size` CHANNELS (cols),
        each group getting its own scale/zero-point.
    group_size: positive int; divides seq_len (for K) and channels (for V).
    bits: bit width (default 4, i.e. 16 levels).

    Per group with values x:
        qmax  = 2**bits - 1
        scale = (max(x) - min(x)) / qmax
        zero  = clip(round(-min(x) / scale), 0, qmax)
        code  = clip(round(x / scale) + zero, 0, qmax)
        x_hat = (code - zero) * scale
    (a constant group reconstructs itself exactly.)

    Returns (K_hat, V_hat), same shapes as K, V.
    """
    raise NotImplementedError('your code here')
