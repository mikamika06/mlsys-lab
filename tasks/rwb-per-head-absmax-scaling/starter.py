def per_head_absmax_e4m3(k: list[list[list[float]]]) -> list[list[list[float]]]:
    """Per-head absmax-scaled fp8 E4M3 quantize-then-dequantize.

    k: (heads, seq, head_dim) float64 array.

    For each head h independently:
      scale_h = max(|k[h]|) / 448.0  (E4M3 max finite magnitude); 1.0 if
        the head is all zeros.
      k_hat[h] = round_to_nearest_e4m3(k[h] / scale_h) * scale_h
        (magnitude rounded to the nearest representable E4M3 grid point,
        sign preserved, magnitude clipped to 448 before rounding).

    Returns the dequantized (heads, seq, head_dim) array.
    """
    raise NotImplementedError('your code here')
