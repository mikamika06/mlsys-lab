import math
import numpy as np

def predict_error_curve(A: np.ndarray, T: int) -> np.ndarray:
    n = A.shape[0]
    mat = [[float(A[i, j]) for j in range(n)] for i in range(n)]
    
    for _ in range(1000):
        max_val = 0.0
        p = 0
        q = 1
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(mat[i][j])
                if val > max_val:
                    max_val = val
                    p = i
                    q = j
                    
        if max_val < 1e-12:
            break
            
        app = mat[p][p]
        aqq = mat[q][q]
        apq = mat[p][q]
        
        if apq != 0.0:
            tau = (aqq - app) / (2.0 * apq)
            if tau >= 0.0:
                t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
            else:
                t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c
        else:
            c = 1.0
            s = 0.0
            t = 0.0
            
        for i in range(n):
            if i != p and i != q:
                api = mat[p][i]
                aqi = mat[q][i]
                mat[p][i] = c * api - s * aqi
                mat[i][p] = mat[p][i]
                mat[q][i] = s * api + c * aqi
                mat[i][q] = mat[q][i]
                
        mat[p][p] = app - t * apq
        mat[q][q] = aqq + t * apq
        mat[p][q] = 0.0
        mat[q][p] = 0.0
        
    vals = [mat[i][i] for i in range(n)]
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if abs(vals[j]) < abs(vals[j + 1]):
                vals[j], vals[j + 1] = vals[j + 1], vals[j]
                
    lam1 = vals[0] if n > 0 else 0.0
    if n > 1:
        lam2 = vals[1]
    else:
        lam2 = 0.0
        
    r = abs(lam2 / lam1) if lam1 != 0 else 0.0
    
    return np.array([r ** t for t in range(T)], dtype=np.float64)
