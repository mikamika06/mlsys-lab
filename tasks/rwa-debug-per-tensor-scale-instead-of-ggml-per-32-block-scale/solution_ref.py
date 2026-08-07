def q4_0_dequantize(x: list[float]) -> list[float]:
    out = [0.0] * len(x)
    for start in range(0, len(x), 32):
        block = x[start:start + 32]
        max_abs = 0.0
        for v in block:
            abs_v = v if v >= 0 else -v
            if abs_v > max_abs:
                max_abs = abs_v
        scale = max_abs / 7.0
        if scale == 0.0:
            for i in range(len(block)):
                out[start + i] = 0.0
        else:
            for i, v in enumerate(block):
                val = round(v / scale)
                if val < -8:
                    val = -8
                elif val > 7:
                    val = 7
                out[start + i] = val * scale
    return out
