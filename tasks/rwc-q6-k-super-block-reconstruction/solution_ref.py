import numpy as np


def q6_k_dequantize(d: float, scales: np.ndarray, ql: np.ndarray, qh: np.ndarray) -> np.ndarray:
    d = float(d)
    scales = np.asarray(scales, dtype=np.int32)
    ql = np.asarray(ql, dtype=np.int32)
    qh = np.asarray(qh, dtype=np.int32)

    y = np.empty(256, dtype=np.float64)
    l = np.arange(32)

    for half in range(2):
        y_off = half * 128
        qlb = ql[half * 64:(half + 1) * 64]
        qhb = qh[half * 32:(half + 1) * 32]
        scb = scales[half * 8:(half + 1) * 8]

        is_ = l // 16

        q1 = ((qlb[l] & 0xF) | (((qhb[l] >> 0) & 3) << 4)) - 32
        q2 = ((qlb[l + 32] & 0xF) | (((qhb[l] >> 2) & 3) << 4)) - 32
        q3 = ((qlb[l] >> 4) | (((qhb[l] >> 4) & 3) << 4)) - 32
        q4 = ((qlb[l + 32] >> 4) | (((qhb[l] >> 6) & 3) << 4)) - 32

        y[y_off + l] = d * scb[is_] * q1
        y[y_off + l + 32] = d * scb[is_ + 2] * q2
        y[y_off + l + 64] = d * scb[is_ + 4] * q3
        y[y_off + l + 96] = d * scb[is_ + 6] * q4

    return y.astype(np.float32)
