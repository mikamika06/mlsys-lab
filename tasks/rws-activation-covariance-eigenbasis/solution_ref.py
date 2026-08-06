import math
import numpy as np

def cov_and_eig(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the empirical covariance matrix of X and its eigen decomposition.
    The eigenvalues are sorted in descending order; eigenvectors correspondingly.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    d = X.shape[1]
    C = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += X[k, i] * X[k, j]
            C[i, j] = s / n

    A_mat = [[C[r, c] for c in range(d)] for r in range(d)]
    V_mat = [[1.0 if r == c else 0.0 for c in range(d)] for r in range(d)]
    
    for _ in range(100):
        max_val = 0.0
        p, q = 0, 1
        for i in range(d):
            for j in range(i + 1, d):
                val = A_mat[i][j]
                if val < 0.0:
                    val = -val
                if val > max_val:
                    max_val = val
                    p, q = i, j
        
        if max_val < 1e-15:
            break
            
        app = A_mat[p][p]
        aqq = A_mat[q][q]
        apq = A_mat[p][q]
        
        if math.fabs(apq) < 1e-15:
            continue
            
        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0.0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
            
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        
        app_old = app
        aqq_old = aqq
        apq_old = apq
        
        A_mat[p][p] = c * c * app_old - 2.0 * c * s * apq_old + s * s * aqq_old
        A_mat[q][q] = s * s * app_old + 2.0 * c * s * apq_old + c * c * aqq_old
        A_mat[p][q] = 0.0
        A_mat[q][p] = 0.0
        
        for k in range(d):
            if k != p and k != q:
                akp_old = A_mat[k][p]
                akq_old = A_mat[k][q]
                A_mat[k][p] = c * akp_old - s * akq_old
                A_mat[p][k] = A_mat[k][p]
                A_mat[k][q] = s * akp_old + c * akq_old
                A_mat[q][k] = A_mat[k][q]
                
        for k in range(d):
            vkp_old = V_mat[k][p]
            vkq_old = V_mat[k][q]
            V_mat[k][p] = c * vkp_old - s * vkq_old
            V_mat[k][q] = s * vkp_old + c * vkq_old

    w = np.zeros(d, dtype=np.float64)
    for i in range(d):
        w[i] = A_mat[i][i]
        
    v = np.zeros((d, d), dtype=np.float64)
    for r in range(d):
        for c in range(d):
            v[r, c] = V_mat[r][c]

    w_list = [w[i] for i in range(d)]
    v_cols = [[v[r, c] for r in range(d)] for c in range(d)]

    for i in range(d - 1):
        max_idx = i
        for j in range(i + 1, d):
            if w_list[j] > w_list[max_idx]:
                max_idx = j
        if max_idx != i:
            w_list[i], w_list[max_idx] = w_list[max_idx], w_list[i]
            v_cols[i], v_cols[max_idx] = v_cols[max_idx], v_cols[i]

    w_out = np.zeros(d, dtype=np.float64)
    for i in range(d):
        w_out[i] = w_list[i]

    v_out = np.zeros((d, d), dtype=np.float64)
    for c in range(d):
        for r in range(d):
            v_out[r, c] = v_cols[c][r]

    return C, w_out, v_out
