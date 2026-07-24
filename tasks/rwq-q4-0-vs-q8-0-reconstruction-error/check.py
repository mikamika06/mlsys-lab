import numpy as np

BLOCK = 32


def _quant_block_rows(W, qmax):
    amax = np.max(np.abs(W), axis=1)
    d = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(W / d[:, None]), -qmax, qmax)
    return codes * d[:, None]


def _oracle(W):
    q4 = _quant_block_rows(W, 8)    # Q4_0: signed 4-bit, range [-8, 7]
    q8 = _quant_block_rows(W, 127)  # Q8_0: signed 8-bit, range [-127, 127]
    mse4 = float(np.mean((q4 - W) ** 2))
    mse8 = float(np.mean((q8 - W) ** 2))
    return mse4, mse8


def grade(sol, fx) -> dict:
    """
    Runs the reference ggml-style block quantization (block size 32, one
    block per row: Q4_0 = signed 4-bit symmetric absmax, Q8_0 = signed
    8-bit symmetric absmax) on the fixed weight fixture with a NumPy
    oracle, and compares the submission's reported reconstruction MSE for
    each format (relative error) plus whether Q8_0 actually beats Q4_0.
    """
    W = np.asarray(fx["gguf_w"], dtype=np.float64)
    mse4_exp, mse8_exp = _oracle(W)

    try:
        mse4_got, mse8_got = sol.q4_q8_reconstruction_mse(W.copy())
        mse4_got = float(mse4_got)
        mse8_got = float(mse8_got)
    except Exception:
        return {"q4_rel_err": float("inf"), "q8_rel_err": float("inf"), "q8_beats_q4": 0.0}

    q4_rel = abs(mse4_got - mse4_exp) / (abs(mse4_exp) + 1e-12)
    q8_rel = abs(mse8_got - mse8_exp) / (abs(mse8_exp) + 1e-12)
    beats = 1.0 if mse8_got < mse4_got else 0.0

    return {"q4_rel_err": q4_rel, "q8_rel_err": q8_rel, "q8_beats_q4": beats}
