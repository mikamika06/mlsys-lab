def hbm_bytes_ratio(N: int, d: int, B: int) -> float:
    """
    Compute the ratio of HBM bytes moved by naïve attention to that of FlashAttention.
    Parameters are integers; the result is returned as a Python float.
    """
    N = int(N)
    d = int(d)
    B = int(B)

    # Bytes for naive implementation
    naive_bytes = 5 * N * d * 4 + 5 * N * N * 4

    # Number of blocks processed by FlashAttention
    num_blocks = (N + B - 1) // B

    # Bytes for FlashAttention per block
    flash_per_block = (
        3 * N * B * 4          # scores write, softmax read/write, multiply read
        + 3 * B * d * 4        # K tile load, V tile load, V tile read
        + N * d * 4            # output block write per block
    )

    flash_bytes = N * d * 4 + num_blocks * flash_per_block

    return float(naive_bytes / flash_bytes)
