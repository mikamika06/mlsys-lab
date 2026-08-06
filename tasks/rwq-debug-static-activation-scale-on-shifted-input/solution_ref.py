import numpy as np


def quantized_linear_dynamic(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    max_abs = 0.0
    shape_x = x.shape
    if len(shape_x) == 2:
        M, K = shape_x
        for i in range(M):
            for j in range(K):
                val = abs(x[i, j])
                if val > max_abs:
                    max_abs = val
    elif len(shape_x) == 1:
        for i in range(shape_x[0]):
            val = abs(x[i])
            if val > max_abs:
                max_abs = val
    else:
        for idx in np.ndindex(shape_x):
            val = abs(x[idx])
            if val > max_abs:
                max_abs = val

    scale = max_abs / 127.0

    if scale == 0.0:
        if len(shape_x) == 2:
            x_hat_list = [[0.0 for _ in range(shape_x[1])] for _ in range(shape_x[0])]
        elif len(shape_x) == 1:
            x_hat_list = [0.0 for _ in range(shape_x[0])]
        else:
            def make_zeros(shape, dim=0):
                if dim == len(shape) - 1:
                    return [0.0 for _ in range(shape[dim])]
                return [make_zeros(shape, dim + 1) for _ in range(shape[dim])]
            x_hat_list = make_zeros(shape_x)
    else:
        if len(shape_x) == 2:
            M, K = shape_x
            x_hat_list = []
            for i in range(M):
                row = []
                for j in range(K):
                    q = round(x[i, j] / scale)
                    if q < -127:
                        q = -127
                    elif q > 127:
                        q = 127
                    row.append(q * scale)
                x_hat_list.append(row)
        elif len(shape_x) == 1:
            K = shape_x[0]
            x_hat_list = []
            for j in range(K):
                q = round(x[j] / scale)
                if q < -127:
                    q = -127
                elif q > 127:
                    q = 127
                x_hat_list.append(q * scale)
        else:
            def compute_q(shape, idxs=()):
                if len(idxs) == len(shape):
                    q = round(x[idxs] / scale)
                    if q < -127:
                        q = -127
                    elif q > 127:
                        q = 127
                    return q * scale
                res = []
                for i in range(shape[len(idxs)]):
                    res.append(compute_q(shape, idxs + (i,)))
                return res
            x_hat_list = compute_q(shape_x)

    x_hat = np.asarray(x_hat_list, dtype=np.float64)

    M = x_hat.shape[0]
    K = x_hat.shape[1]
    N = W.shape[0]

    result_list = []
    for i in range(M):
        row_res = []
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += x_hat[i, k] * W[j, k]
            acc += b[j]
            row_res.append(acc)
        result_list.append(row_res)

    return np.asarray(result_list, dtype=np.float64)
