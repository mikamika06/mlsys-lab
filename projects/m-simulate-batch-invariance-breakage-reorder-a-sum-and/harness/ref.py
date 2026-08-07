import numpy as np

np.random.seed(42)
SIMULATE_INPUTS = [np.random.randn(32, 32).astype(np.float32) for _ in range(3)]

def simulate_reordered_sum(matrix):
    flat = matrix.flatten()
    s1 = float(np.sum(flat))
    idx = np.argsort(np.abs(flat))
    s2 = float(np.sum(flat[idx]))
    return {"standard": s1, "reordered": s2, "delta": abs(s1 - s2)}

CLASSIFY_CASES = [
    {"id": i, "desc": f"case {i}", "category": ("numeric_drift" if i % 4 == 0 else "sampling_difference" if i % 4 == 1 else "template_change" if i % 4 == 2 else "bug")}
    for i in range(12)
]

def classify_diffs(cases):
    return [c["category"] for c in cases]
