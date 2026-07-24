def arithmetic_intensity(num_layers: int,
                         hidden_size: int,
                         num_heads: int,
                         seq_len: int) -> float:
    """
    Compute the arithmetic intensity (FLOPs per byte) for a single‑token decode.
    The formula is derived in the task description and uses only floating‑point
    arithmetic.  All intermediate values are kept as Python floats to avoid
    integer truncation.
    """
    H = hidden_size
    N_h = num_heads
    S = seq_len

    flops_per_layer = 12 * H ** 2 + 2 * H * S
    bytes_per_layer = 48 * H ** 2 + (8 * S * H) / N_h + 4 * H

    return flops_per_layer / bytes_per_layer
