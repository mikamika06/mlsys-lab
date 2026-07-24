import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_error = 0.0
    test_cases = [
        {
            "A_shape": (3,4),
            "B_shape": (4,5),
            "C_shape": (3,5),
            "alpha": 1.0,
            "beta": 1.0,
            "transA": False,
            "transB": False
        },
        {
            "A_shape": (4,3),
            "B_shape": (4,5),
            "C_shape": (1,5),
            "alpha": 1.0,
            "beta": 1.0,
            "transA": True,
            "transB": False
        },
        {
            "A_shape": (3,4),
            "B_shape": (5,4),
            "C_shape": None,
            "alpha": 1.0,
            "beta": 1.0,
            "transA": False,
            "transB": True
        },
        {
            "A_shape": (4,3),
            "B_shape": (5,4),
            "C_shape": (3,5),
            "alpha": 2.0,
            "beta": 0.5,
            "transA": True,
            "transB": True
        },
        {
            "A_shape": (3,4),
            "B_shape": (4,5),
            "C_shape": None,
            "alpha": 3.0,
            "beta": 1.0,
            "transA": False,
            "transB": False
        },
        {
            "A_shape": (2,3),
            "B_shape": (3,4),
            "C_shape": (2,1),
            "alpha": 1.0,
            "beta": 2.0,
            "transA": False,
            "transB": False
        }
    ]

    for case in test_cases:
        A = rng.standard_normal(case["A_shape"])
        B = rng.standard_normal(case["B_shape"])
        C = None if case["C_shape"] is None else rng.standard_normal(case["C_shape"])

        A_mat = A.T if case["transA"] else A
        B_mat = B.T if case["transB"] else B
        Y_ref = case["alpha"] * (A_mat @ B_mat)
        if C is not None:
            Y_ref += case["beta"] * np.broadcast_to(C, Y_ref.shape)

        try:
            Y_sol = sol.gemm(A, B, C=C,
                             alpha=case["alpha"],
                             beta=case["beta"],
                             transA=case["transA"],
                             transB=case["transB"])
        except Exception:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(Y_ref, Y_sol)
        if err > max_error:
            max_error = err

    return {"max_abs_err": max_error}
