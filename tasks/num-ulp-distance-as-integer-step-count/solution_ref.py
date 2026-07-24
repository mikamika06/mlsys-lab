import numpy as np


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def ordered(x):
        bits = x.view(np.uint32)
        sign = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            sign,
            np.bitwise_not(bits),
            bits ^ np.uint32(0x80000000),
        ).astype(np.int64)

    return np.abs(ordered(a) - ordered(b)).astype(np.uint32)
