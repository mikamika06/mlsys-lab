def gemm(A: list[list[float]],
         B: list[list[float]],
         C: list[list[float]] | list[float] | None = None,
         alpha: float = 1.0,
         beta: float = 1.0,
         transA: bool = False,
         transB: bool = False) -> list[list[float]]:
    m = len(A[0]) if transA else len(A)
    k = len(A) if transA else len(A[0])
    n = len(B) if transB else len(B[0])

    result: list[list[float]] = []
    for i in range(m):
        row: list[float] = []
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
