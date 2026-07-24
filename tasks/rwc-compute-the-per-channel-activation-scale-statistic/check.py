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
        X = np.random.randn(*shape).astype(np.float64)
        try:
            got = sol.compute_activation_scale(X)
        except Exception as e:
            # If the function raises an exception, treat error as infinite
            return {"max_abs_err": float("inf")}
        ref = np.mean(np.abs(X), axis=(0, 1))
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        errors.append(float(np.max(np.abs(got - ref))))
    # Return the worst error across all test cases
    return {"max_abs_err": max(errors)}
