def blockwise_absmax(w: list[float], block_size: int = 64) -> list[float]:
    """Return per-block maximum absolute value of w."""
    n = len(w)
    pad_len = (-n) % block_size
    w_padded = list(w) + [0.0] * pad_len
    num_blocks = len(w_padded) // block_size
    res = []
    for i in range(num_blocks):
        m = 0.0
        start = i * block_size
        end = start + block_size
        for j in range(start, end):
            val = abs(w_padded[j])
            if val > m:
                m = val
        res.append(float(m))
    return res
