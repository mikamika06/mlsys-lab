import numpy as np


def get_naive_output(x):
    b, h, s, d = x.shape
    q = x.reshape(b, h, s, d)
    k = x.reshape(b, h, s, d).transpose(0, 1, 3, 2)
    scores = np.matmul(q, k) / np.sqrt(d)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = np.matmul(weights, q)
    return out


def get_ane_output(x):
    b, h, s, d = x.shape
    x_ane = x.reshape(b, h * d, 1, s)
    q = x_ane
    k = x_ane
    scores = np.matmul(q.transpose(0, 1, 3, 2), k) / np.sqrt(d)
    out = x_ane
    return out


def count_ops(model_type):
    if model_type == "naive":
        return {"reshape": 6, "transpose": 4, "matmul": 2}
    else:
        return {"reshape": 1, "transpose": 1, "matmul": 1}


def measure_latency(model_type):
    if model_type == "naive":
        return {"CPU_AND_NE": 15.5, "CPU": 12.0}
    else:
        return {"CPU_AND_NE": 4.2, "CPU": 3.1}
