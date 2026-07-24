import numpy as np


def _fp8_values():
    values = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            value = sign * (mant / 8.0) * (2.0 ** -6)
        elif exp == 15:
            value = sign * (240.0 + mant * 16.0)
        else:
            value = sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        values.append(np.float32(value))
    return values


_FP8_TABLE = _fp8_values()


def _ref_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def _ref_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (((bits >> 16) << 16).view(np.float32))


def _ref_fp8(x):
    arr = np.asarray(x, dtype=np.float32)
    out = []
    for v in arr.ravel():
        best = 0
        best_err = float("inf")
        for code, candidate in enumerate(_FP8_TABLE):
            err = abs(float(v) - float(candidate))
            if err < best_err:
                best_err = err
                best = code
        out.append(_FP8_TABLE[best])
    return np.asarray(out, dtype=np.float32).reshape(arr.shape)


def grade(sol, fx) -> dict:
    x = np.array(
        [
            -500.0,
            -448.0,
            -12.75,
            -1.1,
            -0.0,
            0.0,
            0.1,
            1.0,
            3.14159,
            100.0,
            500.0,
        ],
        dtype=np.float32,
    )

    result = {}
    checks = [
        ("fp16_max_abs_err", _ref_fp16(x), sol.fp16_roundtrip(x)),
        ("bf16_max_abs_err", _ref_bf16(x), sol.bf16_roundtrip(x)),
        ("fp8_max_abs_err", _ref_fp8(x), sol.fp8_e4m3_roundtrip(x)),
    ]

    for name, ref, got in checks:
        try:
            result[name] = float(np.max(np.abs(ref - np.asarray(got, dtype=np.float32))))
        except Exception:
            result[name] = float("inf")
    return result
