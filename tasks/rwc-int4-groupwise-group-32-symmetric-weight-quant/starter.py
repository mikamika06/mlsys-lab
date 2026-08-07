def int4_groupwise_quant(W: list[list[float]], group_size: int=32) -> tuple[list[list[int]], list[list[float]]]:
    """
    W: (rows, cols) weight matrix; `cols` must be a multiple of
        `group_size`.

    Symmetric int4 quantization, applied independently per row and per
    contiguous group of `group_size` values along the columns:

        amax  = max(abs(group))
        scale = amax / 8              (1.0 if amax == 0, to avoid /0)
        code  = clip(round(group / scale), -8, 7)

    Returns (codes, scales):
      codes: (rows, cols) int array, values in [-8, 7].
      scales: (rows, cols // group_size) float array, one scale per row
        per group.
    """
    raise NotImplementedError('your code here')
