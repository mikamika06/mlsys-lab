def nf4_block_absmax_scales(W: list[list[float]]) -> list[float]:
    w = []
    for row in W:
        for val in row:
            w.append(val)

    num_blocks = len(w) // 64
    scales = []
    for i in range(num_blocks):
        block_start = i * 64
        max_val = abs(w[block_start])
        for j in range(1, 64):
            val = abs(w[block_start + j])
            if val > max_val:
                max_val = val
        scales.append(max_val)
    return scales
