import numpy as np


def decode_access_count(shape, layout):
    b, h, s, d = shape

    if layout == "contiguous":
        arr = np.empty(shape, dtype=np.float16)
    elif layout == "strided":
        base = np.empty((b, s, h, d), dtype=np.float16)
        arr = base.transpose(0, 2, 1, 3)
    else:
        raise ValueError("unknown layout")

    itemsize = arr.dtype.itemsize
    strides = tuple(x // itemsize for x in arr.strides)

    lines = set()
    for si in range(s):
        for di in range(d):
            offset = si * strides[2] + di * strides[3]
            lines.add((offset * itemsize) // 64)

    return len(lines)
