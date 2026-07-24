import numpy as np


def _oracle(packed, scales, zero_points, shape, group_size):
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float64)
    zero_points = np.asarray(zero_points, dtype=np.int64)

    n = int(np.prod(shape))
    q = np.empty(n, dtype=np.int64)
    for i in range(n):
        byte = int(packed[i // 2])
        if i % 2 == 0:
            q[i] = byte & 15
        else:
            q[i] = (byte >> 4) & 15

    groups = np.arange(n) // group_size
    out = (q - zero_points[groups]) * scales[groups]
    return out.astype(np.float64).reshape(shape)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0x21, 0xF3], dtype=np.uint8),
            np.array([0.5], dtype=np.float64),
            np.array([0], dtype=np.int64),
            (4,),
            4,
        ),
        (
            np.array([0x01, 0xAB, 0x4F, 0x80], dtype=np.uint8),
            np.array([0.25, 1.5], dtype=np.float64),
            np.array([1, 7], dtype=np.int64),
            (8,),
            4,
        ),
        (
            np.array([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC], dtype=np.uint8),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            np.array([0, 3, 8], dtype=np.int64),
            (3, 4),
            4,
        ),
    ]

    best = 0.0
    for packed, scales, zero_points, shape, group_size in cases:
        ref = _oracle(packed, scales, zero_points, shape, group_size)
        try:
            got = np.asarray(
                sol.dequantize_int4(
                    packed,
                    scales,
                    zero_points,
                    shape,
                    group_size,
                ),
                dtype=np.float64,
            )
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        best = max(best, err)
    return {"max_abs_err": best}
