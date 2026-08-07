NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def nf4_blockwise_dequant(w: list[float], block_size: int = 64) -> list[float]:
    """
    Quantize-then-dequantize `w` through the NF4-blockwise codec: per-block
    absmax scale + nearest-level lookup in the fixed 16-value NF4 codebook.
    """
    xhat = []
    for i in range(0, len(w), block_size):
        block = w[i : i + block_size]

        max_val = 0.0
        for val in block:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
        scale = max_val if max_val != 0.0 else 1.0

        for val in block:
            normalized = val / scale

            best_idx = 0
            min_diff = -1.0
            for idx, level in enumerate(NF4_LEVELS):
                diff = normalized - level
                if diff < 0.0:
                    diff = -diff
                if min_diff < 0.0 or diff < min_diff:
                    min_diff = diff
                    best_idx = idx

            dequantized = NF4_LEVELS[best_idx] * scale
            xhat.append(dequantized)

    return xhat
