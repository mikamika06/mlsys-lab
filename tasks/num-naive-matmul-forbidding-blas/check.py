import inspect
import numpy as np

def _ref(A, B):
    """Naive reference implementation using triple loop."""
    m, p = A.shape
    p2, n = B.shape
    assert p == p2, "Incompatible shapes"
    C = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def grade(sol, fx) -> dict:
    # Compute reference result
    try:
        ref_func = _ref
    except Exception as e:
        return {"max_abs_err": 1.0, "blas_used": 1}
    # Find user's matmul function
    if not hasattr(sol, "matmul"):
        return {"max_abs_err": 1.0, "blas_used": 1}
    user_func = getattr(sol, "matmul")
    # Generate random test matrices
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 7))
    B = rng.standard_normal((7, 4))
    try:
        C_user = user_func(A.astype(np.float64), B.astype(np.float64))
    except Exception:
        return {"max_abs_err": 1.0, "blas_used": 1}
    # Reference
    C_ref = ref_func(A, B)
    max_err = float(np.max(np.abs(C_user - C_ref)))
    # Check BLAS usage by inspecting source code
    try:
        src = inspect.getsource(sol)
    except Exception:
        src = ""
    blas_used = 0
    for token in ("np.dot", "einsum", "@"):
        if token in src:
            blas_used = 1
            break
    return {"max_abs_err": max_err, "blas_used": blas_used}
