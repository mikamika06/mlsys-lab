import random

def gemm_ref(A, B, C=None, alpha=1.0, beta=1.0, transA=False, transB=False):
    m = len(A[0]) if transA else len(A)
    k = len(A) if transA else len(A[0])
    n = len(B) if transB else len(B[0])

    result = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = 0.0
            for p in range(k):
                a_val = A[p][i] if transA else A[i][p]
                b_val = B[j][p] if transB else B[p][j]
                acc += float(a_val) * float(b_val)

            val = alpha * acc
            if C is not None:
                if isinstance(C, (int, float)):
                    c_val = float(C)
                elif not isinstance(C[0], list):
                    c_val = float(C[0]) if len(C) == 1 else float(C[j])
                else:
                    r = 0 if len(C) == 1 else i
                    c = 0 if len(C[0]) == 1 else j
                    c_val = float(C[r][c])
                val += beta * c_val

            row.append(val)
        result.append(row)
    return result

def max_abs_err(Y_stu, Y_ref):
    if not isinstance(Y_stu, list) or len(Y_stu) != len(Y_ref):
        return float('inf')
    err = 0.0
    for i in range(len(Y_ref)):
        if not isinstance(Y_stu[i], list) or len(Y_stu[i]) != len(Y_ref[i]):
            return float('inf')
        for j in range(len(Y_ref[i])):
            err = max(err, abs(Y_stu[i][j] - Y_ref[i][j]))
    return err

def grade(sol, fx) -> dict:
    rng = random.Random(42)
    max_err = 0.0

    test_configs = [
        # (m, k, n, transA, transB, alpha, beta, c_shape_type)
        (2, 2, 2, False, False, 1.0, 1.0, 'none'),
        (2, 3, 2, False, True, 2.0, 3.0, '1d'),
        (3, 2, 4, True, False, 0.5, 2.0, '2d_full'),
        (2, 2, 3, True, True, 1.5, 0.0, 'none'),
        (1, 4, 1, False, False, 1.0, 1.0, 'scalar_shape'),
        (3, 3, 3, False, False, -1.0, 0.5, 'col_vec'),
        (4, 2, 3, True, False, 2.5, -1.5, 'row_vec'),
    ]

    for m, k, n, transA, transB, alpha, beta, c_type in test_configs:
        a_rows, a_cols = (k, m) if transA else (m, k)
        b_rows, b_cols = (n, k) if transB else (k, n)

        A = [[rng.uniform(-2.0, 2.0) for _ in range(a_cols)] for _ in range(a_rows)]
        B = [[rng.uniform(-2.0, 2.0) for _ in range(b_cols)] for _ in range(b_rows)]

        C = None
        if c_type == '1d':
            C = [rng.uniform(-2.0, 2.0) for _ in range(n)]
        elif c_type == '2d_full':
            C = [[rng.uniform(-2.0, 2.0) for _ in range(n)] for _ in range(m)]
        elif c_type == 'scalar_shape':
            C = [[rng.uniform(-2.0, 2.0)]]
        elif c_type == 'col_vec':
            C = [[rng.uniform(-2.0, 2.0)] for _ in range(m)]
        elif c_type == 'row_vec':
            C = [[rng.uniform(-2.0, 2.0) for _ in range(n)]]

        Y_ref = gemm_ref(A, B, C, alpha, beta, transA, transB)
        Y_stu = sol.gemm(A, B, C=C, alpha=alpha, beta=beta, transA=transA, transB=transB)

        err = max_abs_err(Y_stu, Y_ref)
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
