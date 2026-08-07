def per_channel_fp8_quant(W: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Per-row (per-output-channel) E4M3 quantize/dequantize of a weight
    matrix.

    W: (rows, cols) float64 matrix.

    Returns (scales, W_dequant):
    - scales: (rows,) each row's scale = max(|row|)/448 (1.0 for an
      all-zero row).
    - W_dequant: (rows, cols) reconstruction of W after per-row E4M3
      quantize/dequantize.
    """
    raise NotImplementedError('your code here')
