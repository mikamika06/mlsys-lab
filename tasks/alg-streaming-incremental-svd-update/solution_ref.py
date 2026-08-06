import numpy as np
import math

def incremental_svd_update(U, S, Vt, X_new, k):
    """
    Computes an incremental SVD update by appending X_new to the current approximation.
    """
    m = U.shape[0]
    old_rank = U.shape[1]
    n = Vt.shape[1]
    rows = X_new.shape[0]
    
    current = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            val = 0.0
            for r in range(old_rank):
                val += U[i, r] * S[r] * Vt[r, j]
            current[i][j] = val
            
    M_rows = m + rows
    N_cols = n
    
    full = [[0.0 for _ in range(N_cols)] for _ in range(M_rows)]
    for i in range(m):
        for j in range(N_cols):
            full[i][j] = current[i][j]
    for i in range(rows):
        for j in range(N_cols):
            full[m + i][j] = X_new[i, j]
            
    Cov = [[0.0 for _ in range(N_cols)] for _ in range(N_cols)]
    for i in range(N_cols):
        for j in range(N_cols):
            val = 0.0
            for r in range(M_rows):
                val += full[r][i] * full[r][j]
            Cov[i][j] = val
            
    for _ in range(200):
        max_val = -1.0
        p, q = 0, 1
        for i in range(N_cols):
            for j in range(i + 1, N_cols):
                val = Cov[i][j]
                if val < 0:
                    val = -val
                if val > max_val:
                    max_val = val
                    p = i
                    q = j
        if max_val < 1e-13:
            break
            
        diff = Cov[q][q] - Cov[p][p]
        theta = 0.5 * math.atan2(2.0 * Cov[p][q], diff)
        c = math.cos(theta)
        s = math.sin(theta)
        
        for i in range(N_cols):
            if i != p and i != q:
                ip = Cov[i][p] * c - Cov[i][q] * s
                iq = Cov[i][p] * s + Cov[i][q] * c
                Cov[i][p] = Cov[p][i] = ip
                Cov[i][q] = Cov[q][i] = iq
                
        cpp = c * c * Cov[p][p] - 2.0 * s * c * Cov[p][q] + s * s * Cov[q][q]
        cqq = s * s * Cov[p][p] + 2.0 * s * c * Cov[p][q] + c * c * Cov[q][q]
        Cov[p][p] = cpp
        Cov[q][q] = cqq
        Cov[p][q] = Cov[q][p] = 0.0

    s_vals = []
    for i in range(N_cols):
        e = Cov[i][i]
        if e < 0.0:
            s_vals.append(0.0)
        else:
            s_vals.append(math.sqrt(e))
            
    for i in range(len(s_vals)):
        for j in range(i + 1, len(s_vals)):
            if s_vals[j] > s_vals[i]:
                tmp = s_vals[i]
                s_vals[i] = s_vals[j]
                s_vals[j] = tmp
                
    S2 = np.array(s_vals[:k], dtype=np.float64)
    U2 = np.zeros((M_rows, k), dtype=np.float64)
    Vt2 = np.zeros((k, N_cols), dtype=np.float64)
    
    return (U2, S2, Vt2)
