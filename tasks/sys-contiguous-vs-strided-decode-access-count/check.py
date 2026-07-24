import numpy as np


def _oracle(shape, layout):
    b, h, s, d = shape
    if layout == "contiguous":
        arr = np.empty(shape, dtype=np.float16)
    elif layout == "strided":
        base = np.empty((b, s, h, d), dtype=np.float16)
        arr = base.transpose(0, 2, 1, 3)
    else:
        raise ValueError(layout)

    itemsize = arr.dtype.itemsize
    strides = tuple(x // itemsize for x in arr.strides)

    lines = set()
    for si in range(s):
        for di in range(d):
            element_offset = (
                0 * strides[0]
                + 0 * strides[1]
                + si * strides[2]
                + di * strides[3]
            )
            byte_address = element_offset * itemsize
            lines.add(byte_address // 64)

    return len(lines)


def grade(sol, fx) -> dict:
    cases = [
        ((1, 8, 64, 8), "contiguous"),
        ((1, 8, 64, 8), "strided"),
        ((2, 8, 64, 16), "contiguous"),
        ((2, 8, 64, 16), "strided"),
        ((1, 16, 128, 16), "contiguous"),
        ((1, 16, 128, 16), "strided"),
    ]

    ok = 1.0
    for shape, layout in cases:
        try:
            got = sol.decode_access_count(shape, layout)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(shape, layout):
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
