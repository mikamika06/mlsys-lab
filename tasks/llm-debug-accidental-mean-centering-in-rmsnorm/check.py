import numpy as np

def _reference_rms_norm(x, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

def grade(sol, fx) -> dict:
    # Generate a variety of test cases
    rng = np.random.default_rng(42)
    shapes = [(4, 5), (10, 3), (7, 8)]
    epsilons = [1e-5, 1e-6]
    max_err = 0.0

    for shape in shapes:
        x_np = rng.standard_normal(shape).astype(np.float64)
        x_list = x_np.tolist()
        for eps in epsilons:
            try:
                out = sol.rms_norm(x_list, eps=eps)
            except Exception:
                return {"max_abs_err": float("inf")}

            ref = _reference_rms_norm(x_np, eps)

            if not isinstance(out, list):
                return {"max_abs_err": float("inf")}

            try:
                out_np = np.array(out, dtype=np.float64)
            except Exception:
                return {"max_abs_err": float("inf")}

            if out_np.shape != ref.shape:
                return {"max_abs_err": float("inf")}

            err = float(np.max(np.abs(out_np - ref)))
            max_err = max(max_err, err)

    return {"max_abs_err": max_err}
