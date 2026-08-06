import numpy as np


def gptq_quantize(W: np.ndarray, Hinv: np.ndarray, scales: np.ndarray, bits: int = 4) -> dict:
    """
    Column-sequential GPTQ quantization: quantize column q with a fixed
    per-row scale, then propagate the rounding error into the remaining
    columns using the inverse-Hessian, normalized by the pivot [Hinv]_qq.
    """
    W = np.asarray(W, dtype=np.float64)
    Hinv = np.asarray(Hinv, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)

    r, d = W.shape
    L = 2 ** (bits - 1) - 1

    Wc = W.copy()
    codes = np.zeros((r, d), dtype=np.int64)

    for q in range(d):
        c_q_list = []
        deq_list = []
        err_list = []

        for i in range(r):
            val = Wc[i, q] / scales[i]
            rounded = round(val)
            if rounded < -L:
                clipped = -L
            elif rounded > L:
                clipped = L
            else:
                clipped = rounded

            c_q_list.append(int(clipped))
            codes[i, q] = int(clipped)

            deq_val = clipped * scales[i]
            deq_list.append(deq_val)
            err_val = Wc[i, q] - deq_val
            err_list.append(err_val)
            Wc[i, q] = deq_val

        if q + 1 < d:
            h_qq = Hinv[q, q]
            for i in range(r):
                factor_i = err_list[i] / h_qq
                for j in range(q + 1, d):
                    Wc[i, j] -= factor_i * Hinv[q, j]

    return {"codes": codes, "W_hat": Wc}
