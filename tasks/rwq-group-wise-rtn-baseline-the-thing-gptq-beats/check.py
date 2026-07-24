import numpy as np

from mlsys import scorers

GROUP_SIZE = 128
NBITS = 4


def _oracle(W, group_size, nbits):
    qmax = (1 << (nbits - 1)) - 1  # symmetric int4 -> 7
    d_out, d_in = W.shape
    n_groups = d_in // group_size
    codes = np.empty((d_out, d_in), dtype=np.int64)
    Wq = np.empty((d_out, d_in), dtype=np.float64)
    for g in range(n_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W[:, sl]
        amax = np.max(np.abs(block), axis=1)  # (d_out,)
        scale = np.where(amax > 0, amax / qmax, 1.0)
        c = np.clip(np.round(block / scale[:, None]), -qmax, qmax)
        codes[:, sl] = c.astype(np.int64)
        Wq[:, sl] = c * scale[:, None]
    return codes, Wq


def grade(sol, fx) -> dict:
    """
    Runs the reference per-row, per-GROUP_SIZE-column symmetric int4
    round-to-nearest quantization (a NumPy oracle, no error feedback) and
    compares the submission's integer codes (exact) and dequantized
    reconstruction (max abs error) against it.
    """
    W = np.asarray(fx["gptq_w"], dtype=np.float64)
    codes_exp, Wq_exp = _oracle(W, GROUP_SIZE, NBITS)

    try:
        codes_got, Wq_got = sol.rtn_group_quantize(W.copy(), GROUP_SIZE)
        codes_got = np.asarray(codes_got)
        Wq_got = np.asarray(Wq_got, dtype=np.float64)
    except Exception:
        return {"codes_exact_match": 0.0, "recon_max_abs_err": float("inf")}

    if codes_got.shape != codes_exp.shape:
        codes_ok = 0.0
    else:
        codes_ok = 1.0 if np.array_equal(codes_got, codes_exp) else 0.0

    if Wq_got.shape != Wq_exp.shape:
        recon_err = float("inf")
    else:
        recon_err = scorers.max_abs_err(Wq_exp, Wq_got)

    return {"codes_exact_match": codes_ok, "recon_max_abs_err": recon_err}
