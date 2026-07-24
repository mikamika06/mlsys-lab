import numpy as np

from mlsys import scorers


def _make_weight(rng, n_out=96, n_in=128):
    """Weight whose OUTPUT CHANNELS (rows) span ~3 orders of magnitude."""
    base = rng.standard_normal((n_out, n_in))
    row_gain = 10.0 ** rng.uniform(-3.0, 0.0, size=(n_out, 1))
    return base * row_gain


def _reference(W, n_bits):
    """Symmetric per-OUTPUT-CHANNEL (row) quantization — the NumPy oracle."""
    qmax = 2 ** (n_bits - 1) - 1
    amax = np.max(np.abs(W), axis=1, keepdims=True)
    scale = np.where(amax == 0.0, 1.0, amax / qmax)
    q = np.clip(np.rint(W / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def _fail():
    return {
        "channel_rel_err": float("inf"),
        "channel_rel_err_int4": float("inf"),
        "codes_valid": 0.0,
        "size_ratio": 0.0,
        "ref_channel_rel_err": float("nan"),
    }


def _call(sol, W, n_bits):
    """Run the learner's quantizer; return (W_hat, q, scale, ok)."""
    try:
        q, scale = sol.quantize_per_channel(W, n_bits)
    except Exception:
        return None, None, None, False
    try:
        q = np.asarray(q)
        scale = np.asarray(scale)
        W_hat = q.astype(np.float64) * scale.astype(np.float64)
    except Exception:
        return None, None, None, False
    if W_hat.shape != W.shape or not np.all(np.isfinite(W_hat)):
        return None, None, None, False
    return W_hat, q, scale, True


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    W = _make_weight(rng)

    q_ref, s_ref = _reference(W, 8)
    ref_cre = scorers.channel_rel_err(W, q_ref.astype(np.float64) * s_ref, axis=1)

    W8, q8, s8, ok8 = _call(sol, W, 8)
    if not ok8:
        return _fail()
    W4, q4, s4, ok4 = _call(sol, W, 4)
    if not ok4:
        return _fail()

    try:
        cre8 = float(scorers.channel_rel_err(W, W8, axis=1))
        cre4 = float(scorers.channel_rel_err(W, W4, axis=1))
        sr = float(scorers.size_ratio(W.astype(np.float32), q8, s8.astype(np.float32)))
    except Exception:
        return _fail()

    valid = 1.0
    for q, bits in ((q8, 8), (q4, 4)):
        qmax = 2 ** (bits - 1) - 1
        if q.dtype != np.int8 or q.shape != W.shape or int(np.max(np.abs(q.astype(np.int64)))) > qmax:
            valid = 0.0

    return {
        "channel_rel_err": cre8,
        "channel_rel_err_int4": cre4,
        "codes_valid": valid,
        "size_ratio": sr,
        "ref_channel_rel_err": float(ref_cre),
    }
