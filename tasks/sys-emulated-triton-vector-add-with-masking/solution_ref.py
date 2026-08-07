def emulated_triton_add(a: list[float],
                        b: list[float],
                        block_size: int) -> list[float]:
    """Emulate a Triton vector-add kernel with block processing and boundary masking."""
    N = len(a)
    n_blocks = (N + block_size - 1) // block_size
    output = [0.0] * N

    for pid in range(n_blocks):
        offset = pid * block_size
        width = min(block_size, N - offset)

        mask = [i < width for i in range(block_size)]

        a_padded = [0.0] * block_size
        b_padded = [0.0] * block_size
        for i in range(width):
            a_padded[i] = a[offset + i]
            b_padded[i] = b[offset + i]

        a_tile = [a_padded[i] if mask[i] else 0.0 for i in range(block_size)]
        b_tile = [b_padded[i] if mask[i] else 0.0 for i in range(block_size)]

        c_tile = [a_tile[i] + b_tile[i] for i in range(block_size)]

        for i in range(width):
            output[offset + i] = c_tile[i]

    return output
