import numpy as np


def _oracle_quantize(w: np.ndarray):
    w = np.asarray(w, dtype=np.float32)
    m = float(np.max(np.abs(w))) if w.size else 0.0
    scale = m / 127.0 if m > 0.0 else 1.0
    codes = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return codes, scale


def _test_arrays(rng, fx):
    arrays = [np.asarray(fx["weights"], dtype=np.float32)]
    # a spread of shapes / dynamic ranges, including an all-zero tensor
    # and a tensor with a single dominant outlier.
    arrays.append(np.zeros((6, 6), dtype=np.float32))
    for _ in range(4):
        shape = tuple(int(v) for v in rng.integers(1, 80, size=2))
        magnitude = float(10 ** rng.uniform(-4, 4))
        arrays.append((rng.standard_normal(shape) * magnitude).astype(np.float32))
    outlier = rng.normal(0.0, 0.01, size=(40, 20)).astype(np.float32)
    outlier[0, 0] = 500.0
    arrays.append(outlier)
    return arrays


def grade(sol, fx) -> dict:
    """Compares the candidate's int8 codes against an independently
    computed reference codec (`code_exact_fraction`), and checks that the
    dequantized round-trip error never exceeds the theoretical
    round-to-nearest bound of `scale / 2` by more than a small numerical
    tolerance (`max_abs_err`, reported as `error / (scale / 2)`, so a
    fixed threshold of ~1.0 is meaningful across every array's own scale).
    """
    rng = np.random.default_rng(0)
    arrays = _test_arrays(rng, fx)

    total_elems = 0
    matching_elems = 0
    worst_ratio = 0.0

    for w in arrays:
        ref_codes, ref_scale = _oracle_quantize(w)

        try:
            got_codes, got_scale = sol.quantize_symmetric_int8(w)
            got_codes = np.asarray(got_codes)
        except Exception:
            return {"max_abs_err": float("inf"), "code_exact_fraction": 0.0}

        if got_codes.shape != ref_codes.shape or got_codes.dtype != np.int8:
            return {"max_abs_err": float("inf"), "code_exact_fraction": 0.0}
        if abs(float(got_scale) - ref_scale) > 1e-6 * max(1.0, abs(ref_scale)):
            return {"max_abs_err": float("inf"), "code_exact_fraction": 0.0}

        total_elems += ref_codes.size
        matching_elems += int(np.count_nonzero(got_codes == ref_codes))

        try:
            dq = np.asarray(sol.dequantize_symmetric_int8(got_codes, got_scale), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "code_exact_fraction": 0.0}
        if dq.shape != ref_codes.shape:
            return {"max_abs_err": float("inf"), "code_exact_fraction": 0.0}

        wf = np.asarray(w, dtype=np.float64)
        err = np.max(np.abs(dq - wf)) if wf.size else 0.0
        half_scale = ref_scale / 2.0
        ratio = float(err / half_scale) if half_scale > 0 else float(err)
        worst_ratio = max(worst_ratio, ratio)

    code_exact_fraction = matching_elems / total_elems if total_elems else 0.0
    return {
        "max_abs_err": float(worst_ratio),
        "code_exact_fraction": float(code_exact_fraction),
    }
