def quantize_keys_per_channel(K: list[list[float]], bits: int = 4) -> list[list[float]]:
    if not K or not K[0]:
        return []
    rows = len(K)
    cols = len(K[0])
    levels = 2 ** (bits - 1) - 1

    scales = []
    for c in range(cols):
        max_val = 0.0
        for r in range(rows):
            val = abs(K[r][c])
            if val > max_val:
                max_val = val
        scale = max_val / levels if max_val != 0 else 1.0
        scales.append(scale)

    result = []
    for r in range(rows):
        row_res = []
        for c in range(cols):
            scale = scales[c]
            q = round(K[r][c] / scale)
            row_res.append(q * scale)
        result.append(row_res)
    return result
