import numpy as np


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def ordered(x):
        bits = x.view(np.uint32)
        sign = (bits & np.uint32(0x80000000)) != 0
        if sign:
            res = np.bitwise_not(bits)
        else:
            res = bits ^ np.uint32(0x80000000)
        return int(res.astype(np.int64))

    out = []
    for x, y in zip(a.flat, b.flat):
        out.append(abs(ordered(x) - ordered(y)))

    return np.array(out, dtype=np.uint32).reshape(a.shape)
