import numpy as np

def grade(sol, fx) -> dict:
    # Define a set of random shapes to test
    shapes = [
        (4, 5, 10),
        (2, 3, 8),
        (6, 7, 12)
    ]
    errors = []
    for shape in shapes:
        X_np = np.random.randn(*shape).astype(np.float64)
        X_list = X_np.tolist()
        try:
            got = sol.compute_activation_scale(X_list)
        except Exception as e:
            # If the function raises an exception, treat error as infinite
            return {"max_abs_err": float("inf")}
        ref = np.mean(np.abs(X_np), axis=(0, 1))
        if not isinstance(got, list) or len(got) != ref.shape[0]:
            return {"max_abs_err": float("inf")}
        got_arr = np.array(got, dtype=np.float64)
        errors.append(float(np.max(np.abs(got_arr - ref))))
    # Return the worst error across all test cases
    return {"max_abs_err": max(errors)}
