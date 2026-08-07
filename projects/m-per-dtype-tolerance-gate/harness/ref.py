import numpy as np

EPSILON_MAP = {
    "float32": 1.1920929e-07,
    "float16": 9.765625e-04,
    "bfloat16": 7.8125e-03,
}

TEST_CASES = [
    ("float32", 64),
    ("float32", 1024),
    ("float16", 128),
    ("float16", 4096),
    ("bfloat16", 32),
    ("bfloat16", 2048),
]


def compute_reduction_rtol(dtype_str: str, k_terms: int) -> float:
    eps = EPSILON_MAP.get(dtype_str.lower(), 1e-7)
    return float(np.sqrt(float(k_terms)) * eps * 2.5)


def evaluate_gate(actual: np.ndarray, reference: np.ndarray, dtype_str: str, k_terms: int) -> dict:
    rtol = compute_reduction_rtol(dtype_str, k_terms)
    atol = rtol * 2.0
    diff = np.abs(actual - reference)
    tol_bound = atol + rtol * np.abs(reference)
    passed = bool(np.all(diff <= tol_bound))
    max_diff = float(np.max(diff)) if actual.size > 0 else 0.0
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "max_diff": max_diff,
    }


def bisect_divergence(eager_steps, compiled_steps, initial_input, dtype_str, k_terms):
    low = 0
    high = len(eager_steps) - 1
    divergent_idx = -1
    while low <= high:
        mid = (low + high) // 2
        x_eager = initial_input.copy()
        for i in range(mid + 1):
            x_eager = eager_steps[i](x_eager)

        x_comp = initial_input.copy()
        for i in range(mid + 1):
            x_comp = compiled_steps[i](x_comp)

        gate = evaluate_gate(x_comp, x_eager, dtype_str, k_terms)
        if not gate["passed"]:
            divergent_idx = mid
            high = mid - 1
        else:
            low = mid + 1
    return divergent_idx
