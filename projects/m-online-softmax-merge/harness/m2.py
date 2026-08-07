import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"harness_valid": 0.0, "scaling_valid": 0.0}

    try:
        from onlinesoftmax.harness import (
            compute_rel_err,
            verify_tolerance_bounds,
            analyze_error_vs_seqlen,
        )
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    rng = np.random.default_rng(100)
    a = rng.normal(size=(10, 10))
    b = a + 1e-4

    ref_err = ref.reference_compute_rel_err(a, b)
    try:
        got_err = compute_rel_err(a, b)
    except Exception as e:
        out["_note"] = f"compute_rel_err error: {e}"
        return out

    if not np.isclose(ref_err, got_err, rtol=1e-6):
        out["_note"] = f"compute_rel_err mismatch: got {got_err}, expected {ref_err}"
        return out

    ref_bounds = ref.reference_verify_tolerance_bounds(a, b, rtol=1e-3, atol=1e-3)
    try:
        got_bounds = verify_tolerance_bounds(a, b, rtol=1e-3, atol=1e-3)
    except Exception as e:
        out["_note"] = f"verify_tolerance_bounds error: {e}"
        return out

    if got_bounds.get("passed") != ref_bounds["passed"] or not np.isclose(
        got_bounds.get("max_rel_err", -1.0), ref_bounds["max_rel_err"]
    ):
        out["_note"] = f"verify_tolerance_bounds mismatch: got {got_bounds}, expected {ref_bounds}"
        return out

    out["harness_valid"] = 1.0

    seqlens = [64, 128, 256]
    ref_scaling = ref.reference_analyze_error(16, seqlens, chunk_size=32, seed=42)
    try:
        got_scaling = analyze_error_vs_seqlen(16, seqlens, chunk_size=32, seed=42)
    except Exception as e:
        out["_note"] = f"analyze_error_vs_seqlen error: {e}"
        return out

    if not isinstance(got_scaling, dict):
        out["_note"] = "analyze_error_vs_seqlen must return a dictionary"
        return out

    for n in seqlens:
        if n not in got_scaling or not np.isclose(got_scaling[n], ref_scaling[n], rtol=1e-5):
            out["_note"] = f"Scaling curve mismatch at seqlen {n}"
            return out

    out["scaling_valid"] = 1.0
    return out
