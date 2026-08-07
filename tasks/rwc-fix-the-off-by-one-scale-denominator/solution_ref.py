def affine_quant_dequant(x: list[float], bits: int) -> list[float]:
    """Correct affine quantize-dequantize using (2^bits - 1) denominator."""
    n_levels = (1 << bits) - 1
    x_min = min(x)
    x_max = max(x)
    scale = (x_max - x_min) / n_levels if x_max != x_min else 1.0

    out = []
    for val in x:
        q = round((val - x_min) / scale)
        if q < 0:
            q = 0
        elif q > n_levels:
            q = n_levels
        out.append(q * scale + x_min)
    return out
