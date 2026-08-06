import numpy as np

def generate():
    arr = [1e13] + [1.0] * 100000 + [-1e13]
    arr = np.array(arr, dtype=np.float32)
    np.random.seed(42)
    np.random.shuffle(arr)
    return arr.tolist()

def grade(sol, fx) -> dict:
    arr = generate()
    # Reference float64 sum
    expected = float(np.sum(np.array(arr, dtype=np.float64)))

    # Student's float32 compensated sum
    ans = sol.compensated_sum(arr)

    # Relative error
    rel_err = abs(ans - expected) / max(abs(expected), 1e-15)

    return {
        "rel_err": float(rel_err)
    }
