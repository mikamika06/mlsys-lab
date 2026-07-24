import numpy as np


def _oracle_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 32)
    scales = []
    codes = []
    for block in blocks:
        d = np.max(np.abs(block)) / -8.0
        scales.append(np.float16(d))
        q = np.round(block / d).astype(np.int32)
        q = np.clip(q, -8, 7)
        nib = (q + 8).astype(np.uint8)
        packed = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            packed[i] = nib[2 * i] | (nib[2 * i + 1] << 4)
        codes.append(packed)
    return np.asarray(scales, dtype=np.float16), np.asarray(codes, dtype=np.uint8)


def _dequantize(scales, codes):
    scales = np.asarray(scales, dtype=np.float16)
    codes = np.asarray(codes, dtype=np.uint8)
    out = []
    for scale, row in zip(scales, codes):
        vals = np.empty(32, dtype=np.float32)
        for i, byte in enumerate(row):
            vals[2 * i] = (int(byte & 0x0F) - 8) * np.float32(scale)
            vals[2 * i + 1] = (int(byte >> 4) - 8) * np.float32(scale)
        out.append(vals)
    return np.concatenate(out)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1.5, size=320).astype(np.float32)
    try:
        scales, codes = sol.q4_0_quantize(x)
        got = _dequantize(scales, codes)
    except Exception:
        return {"mse": float("inf")}

    ref_scales, ref_codes = _oracle_quantize(x)
    ref = _dequantize(ref_scales, ref_codes)
    return {"mse": float(np.mean((got - ref) ** 2))}
