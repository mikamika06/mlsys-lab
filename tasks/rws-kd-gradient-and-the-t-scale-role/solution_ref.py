import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    if len(shape) == 1:
        C = shape[0]
        max_val = -float('inf')
        for j in range(C):
            val = x[j]
            if val > max_val:
                max_val = val
        sum_e = 0.0
        exps = [0.0] * C
        for j in range(C):
            val = math.exp(x[j] - max_val)
            exps[j] = val
            sum_e += val
        out = np.empty(shape, dtype=np.float64)
        for j in range(C):
            out[j] = exps[j] / sum_e
        return out
    else:
        *batch_shape, C = shape
        out = np.empty(shape, dtype=np.float64)
        for idx in np.ndindex(*batch_shape):
            max_val = -float('inf')
            for j in range(C):
                val = x[idx + (j,)]
                if val > max_val:
                    max_val = val
            sum_e = 0.0
            exps = [0.0] * C
            for j in range(C):
                val = math.exp(x[idx + (j,)] - max_val)
                exps[j] = val
                sum_e += val
            for j in range(C):
                out[idx + (j,)] = exps[j] / sum_e
        return out


def kd_gradient(student_logits, teacher_logits, labels, T, scale_t2=True):
    student_logits = np.asarray(student_logits, dtype=np.float64)
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)

    pt = _softmax(teacher_logits / T)
    ps = _softmax(student_logits / T)

    grad = (ps - pt) / T
    if scale_t2:
        grad = grad * (T * T)
    return grad
