import numpy as np
import math

def svd_rank(A: np.ndarray, tol: float) -> int:
    m, n = A.shape
    if m < n:
        U = np.zeros((n, m), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                U[j, i] = A[i, j]
        m, n = n, m
    else:
        U = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                U[i, j] = A[i, j]

    max_iter = 100
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dot_ii = 0.0
                dot_jj = 0.0
                dot_ij = 0.0
                for k in range(m):
                    ui = U[k, i]
                    uj = U[k, j]
                    dot_ii += ui * ui
                    dot_jj += uj * uj
                    dot_ij += ui * uj
                
                if abs(dot_ij) > 1e-13 * math.sqrt(max(0.0, dot_ii * dot_jj)):
                    changed = True
                    q = dot_ii - dot_jj
                    v = math.hypot(q, 2.0 * dot_ij)
                    
                    if v == 0.0:
                        continue
                        
                    if q >= 0.0:
                        c = math.sqrt((v + q) / (2.0 * v))
                        if c == 0.0:
                            s = 1.0
                        else:
                            s = dot_ij / (v * c)
                    else:
                        s = math.sqrt((v - q) / (2.0 * v))
                        if dot_ij < 0.0:
                            s = -s
                        if s == 0.0:
                            c = 1.0
                        else:
                            c = dot_ij / (v * s)
                            
                    for k in range(m):
                        ui = U[k, i]
                        uj = U[k, j]
                        U[k, i] = c * ui + s * uj
                        U[k, j] = -s * ui + c * uj
        if not changed:
            break
            
    rank = 0
    for j in range(n):
        norm2 = 0.0
        for i in range(m):
            uj = U[i, j]
            norm2 += uj * uj
        if math.sqrt(norm2) > tol:
            rank += 1
            
    return rank
