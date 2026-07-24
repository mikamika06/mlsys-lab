import math

def power_iteration(A: list[list[float]], n_iter: int) -> list[float]:
    n = len(A)
    v = [1.0] * n
    
    # normalize v0
    norm_v = math.sqrt(sum(x*x for x in v))
    v = [x/norm_v for x in v]
    
    for _ in range(n_iter):
        # A * v
        v_next = [0.0] * n
        for i in range(n):
            for j in range(n):
                v_next[i] += A[i][j] * v[j]
        
        # normalize
        norm_v = math.sqrt(sum(x*x for x in v_next))
        v = [x/norm_v for x in v_next]
        
    return v
