import numpy as np

def grade(sol, fx) -> dict:
    np.random.seed(42)
    n, m = 5, 8
    a = np.random.randn(n, m)
    b = np.random.randn(m)
    dc = np.random.randn(n, m)

    eps = 1e-5

    # Known-correct forward for the finite-difference reference
    def forward_ref(a, b):
        return a + b

    def loss(a, b):
        return float(np.sum(dc * forward_ref(a, b)))

    # --- student's backward ---
    try:
        c, backward = sol.broadcast_add(a, b)
        da_student, db_student = backward(dc)
        da_student = np.asarray(da_student, dtype=np.float64)
        db_student = np.asarray(db_student, dtype=np.float64)
    except Exception:
        return {"max_abs_err": 1.0}

    da_ref = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            a_p = a.copy();  a_p[i, j] += eps
            a_m = a.copy();  a_m[i, j] -= eps
            da_ref[i, j] = (loss(a_p, b) - loss(a_m, b)) / (2 * eps)

    db_ref = np.zeros(m)
    for j in range(m):
        b_p = b.copy();  b_p[j] += eps
        b_m = b.copy();  b_m[j] -= eps
        db_ref[j] = (loss(a, b_p) - loss(a, b_m)) / (2 * eps)

    # Shape gate — wrong shape means wrong backward
    if da_student.shape != da_ref.shape or db_student.shape != db_ref.shape:
        return {"max_abs_err": 1.0}

    err_da = float(np.max(np.abs(da_student - da_ref)))
    err_db = float(np.max(np.abs(db_student - db_ref)))
    return {"max_abs_err": max(err_da, err_db)}
