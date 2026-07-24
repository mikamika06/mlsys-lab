import numpy as np


def _group_quant_dequant(x, bits):
    """Uniform affine (asymmetric) min-max quantizer/dequantizer applied
    along the LAST axis: x has shape (..., group_size), and every group
    (the last axis) gets its own scale/zero-point."""
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=-1, keepdims=True)
    xmax = np.max(x, axis=-1, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, qmax)
    return (q - zero_point) * scale


def _oracle(W, s, X, group_size, bits):
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out_f, in_f = W.shape

    Ws = W * s[None, :]
    Ws_grouped = Ws.reshape(out_f, in_f // group_size, group_size)
    Wq_grouped = _group_quant_dequant(Ws_grouped, bits)
    Wq = Wq_grouped.reshape(out_f, in_f)

    Xs = X / s[None, :]
    return Xs @ Wq.T


def _case(seed, out_f, in_f, batch, group_size, bits, s_spread):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((out_f, in_f))
    s = np.exp(rng.uniform(-s_spread, s_spread, size=in_f))
    X = rng.standard_normal((batch, in_f))
    return W, s, X, group_size, bits


def grade(sol, fx) -> dict:
    cases = [
        _case(1, 6, 8, 5, 4, 4, 1.0),
        _case(2, 10, 16, 7, 8, 4, 1.5),
        _case(3, 4, 12, 3, 4, 3, 2.0),
        _case(4, 8, 8, 4, 8, 4, 0.5),
        _case(5, 5, 24, 6, 6, 2, 1.0),
    ]
    # a case with a trivial scale (s == 1 everywhere) -- pipeline should
    # reduce to plain grouped fake-quant of W with no folding effect.
    rng = np.random.default_rng(6)
    W6 = rng.standard_normal((4, 8))
    s6 = np.ones(8)
    X6 = rng.standard_normal((3, 8))
    cases.append((W6, s6, X6, 4, 4))

    worst = 0.0
    for W, s, X, group_size, bits in cases:
        ref = _oracle(W, s, X, group_size, bits)
        try:
            got = np.asarray(
                sol.awq_apply_fixed_scale(
                    W.copy(), s.copy(), X.copy(), group_size, bits
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
