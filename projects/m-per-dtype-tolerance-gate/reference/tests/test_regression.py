import numpy as np
from tolgate.tolerance import compute_reduction_rtol, evaluate_gate
from tolgate.bisection import bisect_divergence


def test_reduction_tolerance_scaling():
    rtol_10 = compute_reduction_rtol("bfloat16", 10)
    rtol_1000 = compute_reduction_rtol("bfloat16", 1000)
    assert rtol_1000 > rtol_10, "rtol must scale with reduction size"


def test_gate_evaluation_pass_and_fail():
    ref_arr = np.ones((10, 10), dtype=np.float32)
    clean_arr = ref_arr + 1e-6
    noisy_arr = ref_arr + 1.0

    res_pass = evaluate_gate(clean_arr, ref_arr, "float32", 128)
    res_fail = evaluate_gate(noisy_arr, ref_arr, "float32", 128)

    assert res_pass["passed"] is True
    assert res_fail["passed"] is False


def test_bisection_pinpoints_faulty_step():
    def step_ok(x):
        return x + 1.0

    def step_bad(x):
        return x * 50.0

    eager = [step_ok, step_ok, step_ok, step_ok]
    compiled = [step_ok, step_ok, step_bad, step_ok]
    init = np.array([1.0, 2.0], dtype=np.float32)

    idx = bisect_divergence(eager, compiled, init, "float32", 10)
    assert idx == 2, f"Expected step index 2, got {idx}"
