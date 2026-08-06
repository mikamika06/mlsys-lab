def read_singular_values_variance_explained(A):
    import numpy as np
    import math

    n, d = A.shape
    k = min(n, d)
    M = np.zeros((k, k), dtype=np.float64)
    
    if n >= d:
        for i in range(k):
            for j in range(k):
                val = 0.0
                for r in range(n):
                    val += A[r, i] * A[r, j]
                M[i, j] = val
    else:
        for i in range(k):
            for j in range(k):
                val = 0.0
                for r in range(d):
                    val += A[i, r] * A[j, r]
                M[i, j] = val

    for sweep in range(50):
        changed = False
        for p in range(k):
            for q in range(p + 1, k):
                if abs(M[p, q]) > 1e-13:
                    changed = True
                    tau = (M[q, q] - M[p, p]) / (2.0 * M[p, q])
                    if tau >= 0:
                        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c
                    
                    M_pp = M[p, p]
                    M_qq = M[q, q]
                    M[p, p] = M_pp - t * M[p, q]
                    M[q, q] = M_qq + t * M[p, q]
                    M[p, q] = 0.0
                    M[q, p] = 0.0
                    
                    for i in range(k):
                        if i != p and i != q:
                            M_ip = M[i, p]
                            M_iq = M[i, q]
                            M[i, p] = c * M_ip - s * M_iq
                            M[p, i] = M[i, p]
                            M[i, q] = s * M_ip + c * M_iq
                            M[q, i] = M[i, q]
        if not changed:
            break

    eigenvalues = np.zeros(k, dtype=np.float64)
    for i in range(k):
        eigenvalues[i] = M[i, i]
        
    for i in range(k):
        for j in range(i + 1, k):
            if eigenvalues[j] > eigenvalues[i]:
                tmp = eigenvalues[i]
                eigenvalues[i] = eigenvalues[j]
                eigenvalues[j] = tmp
                
    S = np.zeros(k, dtype=np.float64)
    for i in range(k):
        if eigenvalues[i] > 0:
            S[i] = math.sqrt(eigenvalues[i])
        else:
            S[i] = 0.0
            
    total = 0.0
    for i in range(k):
        total += S[i] * S[i]
        
    if total == 0.0:
        return np.zeros_like(S, dtype=np.float64)
        
    var_exp = np.zeros(k, dtype=np.float64)
    for i in range(k):
        var_exp[i] = (S[i] * S[i]) / total
        
    res = np.zeros(k, dtype=np.float64)
    cumsum = 0.0
    for i in range(k):
        cumsum += var_exp[i]
        res[i] = cumsum
        
    return res
