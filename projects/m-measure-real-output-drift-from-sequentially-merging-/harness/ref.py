import numpy as np

def get_test_cases():
    rng = np.random.default_rng(1337)
    cases = []
    for _ in range(5):
        w_base = rng.standard_normal((32, 32))
        delta1 = rng.standard_normal((32, 32))
        delta2 = rng.standard_normal((32, 32))
        scale1 = float(rng.uniform(0.1, 2.0))
        scale2 = float(rng.uniform(0.1, 2.0))
        x = rng.standard_normal((10, 32))
        cases.append({
            "w_base": w_base,
            "delta1": delta1,
            "delta2": delta2,
            "scale1": scale1,
            "scale2": scale2,
            "x": x
        })
    return cases

def merge_adapters_ref(w_base, delta1, delta2, scale1=1.0, scale2=1.0):
    return w_base + scale1 * delta1 + scale2 * delta2

def compute_relative_error_ref(output_ref, output_merged):
    diff = np.linalg.norm(output_merged - output_ref)
    denom = np.linalg.norm(output_ref)
    if denom == 0.0:
        return float(diff)
    return float(diff / denom)

def evaluate_drift_ref(w_base, delta1, delta2, x, scale1=1.0, scale2=1.0):
    out_ref = np.dot(x, w_base) + scale1 * np.dot(x, delta1) + scale2 * np.dot(x, delta2)
    w_merged = merge_adapters_ref(w_base, delta1, delta2, scale1, scale2)
    out_merged = np.dot(x, w_merged)
    return compute_relative_error_ref(out_ref, out_merged)
