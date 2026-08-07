def bitwidth_group_mse_frontier(W: list[list[float]], bit_options: list[int], group_size_options: list[int | None]) -> list[list[float]]:
    """Sweep grouped affine quantization over bit widths and group sizes,
    reporting reconstruction MSE for each combination.

    W: (rows, cols) float64 array.
    bit_options: list of bit widths to sweep, e.g. [2, 4].
    group_size_options: list of group sizes to sweep; each entry is either
        a positive int dividing cols evenly (per-row grouping along the
        columns) or None for per-tensor (a single group over all of W).

    Returns a (len(bit_options), len(group_size_options)) array `mse`
    where mse[i, j] is the mean squared error of quantize-then-dequantize
    W with bit_options[i] bits and group size group_size_options[j].
    """
    raise NotImplementedError('your code here')
