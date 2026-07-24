import numpy as np

def grade(sol, fx) -> dict:
    # Reference implementation using NumPy's dot
    def ref_dot(a, b):
        return float(np.dot(a, b))
    
    cases = [
        (np.arange(5, dtype=np.float64), np.arange(5, 10, dtype=np.float64)),
        (np.random.rand(100).astype(np.float64), np.random.rand(100).astype(np.float64)),
        (np.ones(50, dtype=np.float64) * 3.14, np.linspace(-1, 1, 50, dtype=np.float64)),
        (np.array([0., 0., 0.], dtype=np.float64), np.array([5., -2., 7.], dtype=np.float64)),
    ]
    
    max_err = 0.0
    for a, b in cases:
        try:
            got = sol.dot_product(a, b)
        except Exception:
            return {"rel_err": float("inf")}
        ref = ref_dot(a, b)
        err = np.linalg.norm(np.array(got) - np.array(ref)) / (np.linalg.norm(ref) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
