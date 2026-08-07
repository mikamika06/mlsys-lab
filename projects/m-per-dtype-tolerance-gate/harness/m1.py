import numpy as np
import ref


def check(workdir):
    from tolgate.tolerance import compute_reduction_rtol, evaluate_gate

    out = {"tolerance_cases_matched": 0.0, "gating_decisions_matched": 0.0}

    matched_cases = 0
    for dtype_str, k in ref.TEST_CASES:
        want = ref.compute_reduction_rtol(dtype_str, k)
        got = compute_reduction_rtol(dtype_str, k)
        if np.isclose(want, got, rtol=1e-5):
            matched_cases += 1
    out["tolerance_cases_matched"] = float(matched_cases)

    np.random.seed(42)
    ref_arr = np.random.randn(20, 20).astype(np.float32)
    noise = np.random.randn(20, 20).astype(np.float32) * 1e-2
    actual_arr = ref_arr + noise

    want_gate = ref.evaluate_gate(actual_arr, ref_arr, "bfloat16", 512)
    got_gate = evaluate_gate(actual_arr, ref_arr, "bfloat16", 512)

    if (
        isinstance(got_gate, dict)
        and got_gate.get("passed") == want_gate["passed"]
        and np.isclose(got_gate.get("rtol", 0), want_gate["rtol"], rtol=1e-5)
    ):
        out["gating_decisions_matched"] = 1.0

    return out
