import numpy as np


def fused_qkv_projection(X, Wq, Wk, Wv):
    w_qkv = np.concatenate((Wq, Wk, Wv), axis=1)
    qkv = np.matmul(X, w_qkv)
    m = Wq.shape[1]
    return qkv[:, :m], qkv[:, m:2 * m], qkv[:, 2 * m:3 * m]
