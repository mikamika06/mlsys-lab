import numpy as np


def _oracle_scale(W):
    scale = np.max(np.abs(W), axis=1) / 127.0
    scale = np.where(scale == 0.0, 1.0, scale)
    return scale


def _make_case(rng, N, D, C, lo, hi):
    X = rng.standard_normal((N, D)).astype(np.float64)
    row_scales = rng.uniform(lo, hi, size=C)
    W = (rng.standard_normal((C, D)) * row_scales[:, None]).astype(np.float64)
    b = (rng.standard_normal(C) * 0.1).astype(np.float64)
    return X, W, b


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        _make_case(rng, 400, 32, 6, 0.05, 40.0),
        _make_case(rng, 200, 16, 4, 0.1, 20.0),
        _make_case(rng, 300, 48, 8, 1.0, 1.0),  # equal scales edge case
    ]

    worst_agree = 1.0
    quant_valid = 1.0

    for X, W, b in cases:
        ref_logits = X @ W.T + b
        ref_argmax = np.argmax(ref_logits, axis=1)
        ref_scale = _oracle_scale(W)

        try:
            got_logits, got_w_int8, got_scale = sol.quantize_classifier_head(X, W, b)
            got_logits = np.asarray(got_logits, dtype=np.float64)
            got_w_int8 = np.asarray(got_w_int8)
            got_scale = np.asarray(got_scale, dtype=np.float64)
        except Exception:
            return {"argmax_agreement": 0.0, "quant_valid": 0.0}

        if got_logits.shape != ref_logits.shape:
            return {"argmax_agreement": 0.0, "quant_valid": 0.0}

        agree = float(np.mean(np.argmax(got_logits, axis=1) == ref_argmax))
        worst_agree = min(worst_agree, agree)

        # structural checks on the returned quantized weights
        if got_w_int8.shape != W.shape or got_scale.shape != (W.shape[0],):
            quant_valid = 0.0
            continue
        if not np.all(got_w_int8 == np.round(got_w_int8)):
            quant_valid = 0.0
            continue
        if np.any(got_w_int8 < -127) or np.any(got_w_int8 > 127):
            quant_valid = 0.0
            continue

        scale_rel_err = np.abs(got_scale - ref_scale) / np.maximum(np.abs(ref_scale), 1e-12)
        if np.max(scale_rel_err) > 1e-6:
            quant_valid = 0.0
            continue

        recon = X @ (got_w_int8.astype(np.float64) * got_scale[:, None]).T + b
        if np.max(np.abs(recon - got_logits)) > 1e-6 * (np.max(np.abs(recon)) + 1.0):
            quant_valid = 0.0
            continue

    return {"argmax_agreement": worst_agree, "quant_valid": quant_valid}
