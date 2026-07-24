import math

def power_iteration(A: list[list[float]], n_iter: int) -> list[float]:
    # WRONG IMPLEMENTATION: does not normalize properly
    n = len(A)
    v = [1.0] * n
    for _ in range(n_iter):
        v_next = [0.0] * n
        for i in range(n):
            for j in range(n):
                v_next[i] += A[i][j] * v[j]
        v = v_next
    return v
