import numpy as np

EPSILON_MAP = {
    "float32": 1.1920929e-07,
    "float16": 9.765625e-04,
    "bfloat16": 7.8125e-03,
}


def compute_reduction_rtol(dtype_str: str, k_terms: int) -> float:
    """Derive the expected relative tolerance for K-term reduction."""
    eps = EPSILON_MAP.get(dtype_str.lower(), 1e-7)
    return float(np.sqrt(float(k_terms)) * eps * 2.5)


def evaluate_gate(actual: np.ndarray, reference: np.ndarray, dtype_str: str, k_terms: int) -> dict:
    """Evaluate whether actual tensor satisfies tolerance gates against reference."""
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
