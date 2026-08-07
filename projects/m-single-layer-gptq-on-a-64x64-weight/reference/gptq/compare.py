import numpy as np
from gptq.single import run_gptq_single


def compare_mse(W, X, invH, bits=4):
    W_q = run_gptq_single(W, invH, bits=bits)
    qmax = (2 ** (bits - 1)) - 1
    scale = np.max(np.abs(W), axis=0) / qmax
    scale = np.maximum(scale, 1e-8)
    W_rtn = np.clip(np.round(W / scale) * scale, -scale * qmax, scale * qmax)
    out_float = X @ W
    out_gptq = X @ W_q
    out_rtn = X @ W_rtn
    mse_gptq = np.mean((out_float - out_gptq) ** 2)
    mse_rtn = np.mean((out_float - out_rtn) ** 2)
    return mse_gptq, mse_rtn
