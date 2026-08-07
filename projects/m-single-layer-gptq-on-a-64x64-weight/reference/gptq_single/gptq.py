import numpy as np

def compute_hessian_inverse(X, damping=0.01):
    H = X.T @ X + damping * np.eye(X.shape[1])
    return np.linalg.inv(H)

def gptq_quantize(W, X, bits=4, damping=0.01):
    Hinv = compute_hessian_inverse(X, damping)
    W_q = W.copy()
    H_inv = Hinv.copy()
    d = W.shape[1]
    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1
    for i in range(d):
        w = W_q[:, i]
        scale = (w.max() - w.min()) / (q_max - q_min + 1e-8)
        scale = max(scale, 1e-8)
        zero = round(-w.min() / scale + q_min)
        zero = np.clip(zero, q_min, q_max)
        q = np.clip(np.round(w / scale + zero), q_min, q_max)
        w_quant = (q - zero) * scale
        err = (w - w_quant) / H_inv[i, i]
        W_q[:, i:] -= np.outer(err, H_inv[i, i:])
        W_q[:, i] = w_quant
    return W_q

def rtn_quantize(W, bits=4):
    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1
    scale = (W.max() - W.min()) / (q_max - q_min + 1e-8)
    scale = max(scale, 1e-8)
    zero = round(-W.min() / scale + q_min)
    zero = np.clip(zero, q_min, q_max)
    q = np.clip(np.round(W / scale + zero), q_min, q_max)
    return (q - zero) * scale
