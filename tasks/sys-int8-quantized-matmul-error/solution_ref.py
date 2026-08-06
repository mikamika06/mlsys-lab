import math
import numpy as np


def quantized_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def quantize(x):
        max_val = 0.0
        for val in x.flat:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
        scale = max_val / 127.0
        if scale == 0.0:
            scale = 1.0

        q = np.empty(x.shape, dtype=np.int8)
        it_x = np.nditer(x, flags=['multi_index'])
        while not it_x.finished:
            val = it_x[0]
            idx = it_x.multi_index
            div = val / scale
            if div >= 0.0:
                rounded = math.floor(div + 0.5)
            else:
                rounded = math.ceil(div - 0.5)
            
            if rounded < -127:
                clipped = -127
            elif rounded > 127:
                clipped = 127
            else:
                clipped = int(rounded)
            q[idx] = clipped
            it_x.iternext()

        return q, scale

    qA, sA = quantize(A)
    qB, sB = quantize(B)

    m = qA.shape[0]
    k = qA.shape[1]
    n = qB.shape[1]

    acc = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            s = 0
            for l in range(k):
                s += int(qA[i, l]) * int(qB[l, j])
            acc[i, j] = s

    res = np.empty((m, n), dtype=np.float64)
    scale_prod = sA * sB
    for i in range(m):
        for j in range(n):
            res[i, j] = float(acc[i, j]) * scale_prod

    return res
