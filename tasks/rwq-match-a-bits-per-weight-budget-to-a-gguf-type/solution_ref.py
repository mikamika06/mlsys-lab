GGUF_TYPES = [
    ("Q2_K",   16, 64),
    ("Q4_K_M", 32, 64),
    ("Q6_K",   48, 64),
    ("Q8_0",   64, 64),
]

def _compute_bpw() -> list[float]:
    result = []
    for i in range(len(GGUF_TYPES)):
        block_bytes = float(GGUF_TYPES[i][1])
        weights = float(GGUF_TYPES[i][2])
        result.append(8.0 * block_bytes / weights)
    return result

def match_bpw(target_bpw: float) -> int:
    """Return the index of the GGUF type whose bpw is closest to target_bpw."""
    bpw = _compute_bpw()

    diff = []
    for i in range(len(bpw)):
        val = bpw[i] - float(target_bpw)
        if val < 0.0:
            val = -val
        diff.append(val)

    best_idx = 0
    min_val = diff[0]
    for i in range(1, len(diff)):
        if diff[i] < min_val:
            min_val = diff[i]
            best_idx = i

    return int(best_idx)
