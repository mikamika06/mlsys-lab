import numpy as np


def dequant_q6_k_superblock(ql: np.ndarray, qh: np.ndarray, scales: np.ndarray, d: float) -> np.ndarray:
    """Dequantize one GGML Q6_K super-block (256 elements) from its packed
    fields: `ql` (128 bytes, low 4 bits x2 per byte), `qh` (64 bytes, high
    2 bits x4 per byte), `scales` (16 signed sub-block scales), and the
    super-block float scale `d`.
    """
    ql = np.asarray(ql, dtype=np.int64)
    qh = np.asarray(qh, dtype=np.int64)
    sc = np.asarray(scales, dtype=np.int64)
    d = float(d)

    y = np.zeros(256, dtype=np.float64)
    l = np.arange(32)
    is_ = l // 16  # 0 for l < 16, 1 for l >= 16

    for c in range(2):
        qlc = ql[64 * c: 64 * c + 64]
        qhc = qh[32 * c: 32 * c + 32]
        scc = sc[8 * c: 8 * c + 8]

        q1 = (qlc[l] & 0xF | ((qhc[l] >> 0) & 3) << 4) - 32
        q2 = (qlc[l + 32] & 0xF | ((qhc[l] >> 2) & 3) << 4) - 32
        q3 = (qlc[l] >> 4 | ((qhc[l] >> 4) & 3) << 4) - 32
        q4 = (qlc[l + 32] >> 4 | ((qhc[l] >> 6) & 3) << 4) - 32

        yc = np.zeros(128, dtype=np.float64)
        yc[l] = d * scc[is_ + 0] * q1
        yc[l + 32] = d * scc[is_ + 2] * q2
        yc[l + 64] = d * scc[is_ + 4] * q3
        yc[l + 96] = d * scc[is_ + 6] * q4

        y[128 * c: 128 * c + 128] = yc

    return y
