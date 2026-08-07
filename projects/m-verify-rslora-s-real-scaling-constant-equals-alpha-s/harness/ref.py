import numpy as np

def generate_test_cases():
    np.random.seed(42)
    cases = []
    for r in [16, 64, 128, 256]:
        alpha = 32.0
        x = np.random.randn(8, 64).astype(np.float32)
        w_a = np.random.randn(r, 64).astype(np.float32) / np.sqrt(r)
        w_b = np.random.randn(64, r).astype(np.float32) / np.sqrt(r)
        cases.append({
            "rank": r,
            "alpha": alpha,
            "x": x,
            "w_a": w_a,
            "w_b": w_b
        })
    return cases
