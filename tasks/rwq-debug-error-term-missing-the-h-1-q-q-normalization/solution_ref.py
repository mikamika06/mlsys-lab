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
        w_q = Wc[:, q]
        c_q = np.clip(np.round(w_q / scales), -L, L)
        codes[:, q] = c_q.astype(np.int64)
        deq = c_q * scales
        err = w_q - deq
        Wc[:, q] = deq

        if q + 1 < d:
            factor = err / Hinv[q, q]
            Wc[:, q + 1:] -= np.outer(factor, Hinv[q, q + 1:])

    return {"codes": codes, "W_hat": Wc}
