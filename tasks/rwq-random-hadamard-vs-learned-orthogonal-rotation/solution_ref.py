import math
import numpy as np


def _q4(a):
    a = np.asarray(a, dtype=np.float64)
    shape = a.shape
    max_abs = 0.0
    if len(shape) == 2:
        rows, cols = shape
        for i in range(rows):
            for j in range(cols):
                val = a[i, j]
                abs_val = val if val >= 0.0 else -val
                if abs_val > max_abs:
                    max_abs = abs_val
    elif len(shape) == 1:
        for i in range(shape[0]):
            val = a[i]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
    else:
        for val in a.flat:
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val

    scale = max_abs / 7.0
    if scale == 0.0:
        return np.zeros_like(a)

    res_list = []
    if len(shape) == 2:
        for i in range(rows):
            row_list = []
            for j in range(cols):
                val = a[i, j] / scale
                rounded = round(val)
                clipped = -8.0 if rounded < -8.0 else (7.0 if rounded > 7.0 else rounded)
                row_list.append(clipped * scale)
            res_list.append(row_list)
    elif len(shape) == 1:
        for i in range(shape[0]):
            val = a[i] / scale
            rounded = round(val)
            clipped = -8.0 if rounded < -8.0 else (7.0 if rounded > 7.0 else rounded)
            res_list.append(clipped * scale)
    return np.asarray(res_list, dtype=np.float64)


def _hadamard(n):
    H_list = [[1.0]]
    while len(H_list) < n:
        top = [row + row for row in H_list]
        bottom = [row + [-x for x in row] for row in H_list]
        H_list = top + bottom

    scale = math.sqrt(n)
    res_list = [[val / scale for val in row] for row in H_list]
    return np.asarray(res_list, dtype=np.float64)


def _matmul(A, B):
    rA, cA = A.shape
    rB, cB = B.shape
    res = []
    for i in range(rA):
        row = []
        for j in range(cB):
            s = 0.0
            for k in range(cA):
                s += A[i, k] * B[k, j]
            row.append(s)
        res.append(row)
    return np.asarray(res, dtype=np.float64)


def _matmul_T_left(A, B):
    rA, cA = A.shape
    rB, cB = B.shape
    res = []
    for i in range(cA):
        row = []
        for j in range(cB):
            s = 0.0
            for k in range(rA):
                s += A[k, i] * B[k, j]
            row.append(s)
        res.append(row)
    return np.asarray(res, dtype=np.float64)


def _mse(A, B):
    rows, cols = A.shape
    total_sum = 0.0
    count = 0
    for i in range(rows):
        for j in range(cols):
            diff = A[i, j] - B[i, j]
            total_sum += diff * diff
            count += 1
    return total_sum / count


def w4a4_rotation_mse(W, X, R):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    Y = _matmul(W, X)

    H = _hadamard(W.shape[0])
    Y_h = _matmul(_q4(_matmul(W, H)), _q4(_matmul_T_left(H, X)))

    Y_r = _matmul(_q4(_matmul(W, R)), _q4(_matmul_T_left(R, X)))

    return (
        float(_mse(Y_h, Y)),
        float(_mse(Y_r, Y)),
    )
