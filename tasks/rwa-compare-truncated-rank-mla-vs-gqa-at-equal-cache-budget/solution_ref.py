import numpy as np
import math


def _softmax(x, axis=-1):
    shape = x.shape
    out = np.empty(shape, dtype=x.dtype)
    
    if len(shape) == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    max_val = x[i, j, k, 0]
                    for m in range(1, shape[3]):
                        if x[i, j, k, m] > max_val:
                            max_val = x[i, j, k, m]
                    
                    sum_val = 0.0
                    for m in range(shape[3]):
                        e_val = math.exp(x[i, j, k, m] - max_val)
                        out[i, j, k, m] = e_val
                        sum_val += e_val
                        
                    for m in range(shape[3]):
                        out[i, j, k, m] /= sum_val
    elif len(shape) == 2:
        for i in range(shape[0]):
            max_val = x[i, 0]
            for m in range(1, shape[1]):
                if x[i, m] > max_val:
                    max_val = x[i, m]
            
            sum_val = 0.0
            for m in range(shape[1]):
                e_val = math.exp(x[i, m] - max_val)
                out[i, m] = e_val
                sum_val += e_val
                
            for m in range(shape[1]):
                out[i, m] /= sum_val
    return out


def _attention(Q, K, V):
    batch, seq_q, n_heads, d = Q.shape
    seq_k = K.shape[1]
    
    out = np.empty((batch, seq_q, n_heads, d), dtype=Q.dtype)
    scores = np.empty((batch, n_heads, seq_q, seq_k), dtype=Q.dtype)
    
    sqrt_d = math.sqrt(d)
    
    for b in range(batch):
        for h in range(n_heads):
            for i in range(seq_q):
                for j in range(seq_k):
                    s = 0.0
                    for k_idx in range(d):
                        s += Q[b, i, h, k_idx] * K[b, j, h, k_idx]
                    scores[b, h, i, j] = s / sqrt_d
                    
    weights = _softmax(scores, axis=-1)
    
    for b in range(batch):
        for h in range(n_heads):
            for i in range(seq_q):
                for k_idx in range(d):
                    s = 0.0
                    for j in range(seq_k):
                        s += weights[b, h, i, j] * V[b, j, h, k_idx]
                    out[b, i, h, k_idx] = s
                    
    return out


def svd_jacobi_1d(A, max_iter=30, tol=1e-9):
    m, n = A.shape
    V = np.empty((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            V[i, j] = 1.0 if i == j else 0.0
            
    for _ in range(max_iter):
        changed = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = 0.0
                b = 0.0
                c = 0.0
                for k in range(m):
                    a += A[k, i] * A[k, i]
                    b += A[k, j] * A[k, j]
                    c += A[k, i] * A[k, j]
                
                if math.fabs(c) > tol:
                    changed = True
                    zeta = (b - a) / (2.0 * c)
                    t = 1.0 / (math.fabs(zeta) + math.sqrt(1.0 + zeta * zeta))
                    if zeta < 0:
                        t = -t
                    cs = 1.0 / math.sqrt(1.0 + t * t)
                    sn = cs * t
                    
                    for k in range(m):
                        aki = A[k, i]
                        akj = A[k, j]
                        A[k, i] = cs * aki - sn * akj
                        A[k, j] = sn * aki + cs * akj
                        
                    for k in range(n):
                        vki = V[k, i]
                        vkj = V[k, j]
                        V[k, i] = cs * vki - sn * vkj
                        V[k, j] = sn * vki + cs * vkj
        if not changed:
            break
            
    S = np.empty((n,), dtype=A.dtype)
    for i in range(n):
        s2 = 0.0
        for k in range(m):
            s2 += A[k, i] * A[k, i]
        S[i] = math.sqrt(s2)
        
    for i in range(n):
        for j in range(i + 1, n):
            if S[j] > S[i]:
                tmp = S[i]
                S[i] = S[j]
                S[j] = tmp
                for k in range(m):
                    tmp_v = A[k, i]
                    A[k, i] = A[k, j]
                    A[k, j] = tmp_v
                for k in range(n):
                    tmp_v = V[k, i]
                    V[k, i] = V[k, j]
                    V[k, j] = tmp_v
                    
    U = np.empty((m, n), dtype=A.dtype)
    for i in range(n):
        s = S[i]
        if s > 1e-12:
            for k in range(m):
                U[k, i] = A[k, i] / s
        else:
            for k in range(m):
                U[k, i] = 0.0
                
    Vt = np.empty((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            Vt[i, j] = V[j, i]
            
    return U, S, Vt


def mla_gqa_equal_budget_compare(Q: np.ndarray, K: np.ndarray, V: np.ndarray, group_size: int):
    batch, seq_k, n_heads, d = K.shape
    n_kv = n_heads // group_size

    Kg = np.empty((batch, seq_k, n_kv, d), dtype=K.dtype)
    Vg = np.empty((batch, seq_k, n_kv, d), dtype=V.dtype)
    for b in range(batch):
        for s in range(seq_k):
            for i in range(n_kv):
                for j in range(d):
                    k_sum = 0.0
                    v_sum = 0.0
                    for g in range(group_size):
                        h = i * group_size + g
                        k_sum += K[b, s, h, j]
                        v_sum += V[b, s, h, j]
                    Kg[b, s, i, j] = k_sum / group_size
                    Vg[b, s, i, j] = v_sum / group_size

    K_bc = np.empty((batch, seq_k, n_heads, d), dtype=K.dtype)
    V_bc = np.empty((batch, seq_k, n_heads, d), dtype=V.dtype)
    for b in range(batch):
        for s in range(seq_k):
            for i in range(n_kv):
                for g in range(group_size):
                    h = i * group_size + g
                    for j in range(d):
                        K_bc[b, s, h, j] = Kg[b, s, i, j]
                        V_bc[b, s, h, j] = Vg[b, s, i, j]

    gqa_out = _attention(Q, K_bc, V_bc)

    rank = 2 * n_kv * d
    n_features = 2 * n_heads * d
    
    Kr = np.empty((batch, seq_k, n_heads, d), dtype=K.dtype)
    Vr = np.empty((batch, seq_k, n_heads, d), dtype=V.dtype)

    M = np.empty((seq_k, n_features), dtype=K.dtype)
    for b in range(batch):
        for s in range(seq_k):
            idx = 0
            for h in range(n_heads):
                for j in range(d):
                    M[s, idx] = K[b, s, h, j]
                    idx += 1
            for h in range(n_heads):
                for j in range(d):
                    M[s, idx] = V[b, s, h, j]
                    idx += 1
                    
        A = np.empty((seq_k, n_features), dtype=M.dtype)
        for i in range(seq_k):
            for j in range(n_features):
                A[i, j] = M[i, j]
                
        U_mat, S_vec, Vt_mat = svd_jacobi_1d(A)
        r = rank if rank < n_features else n_features
        
        for s in range(seq_k):
            for j in range(n_features):
                val = 0.0
                for k in range(r):
                    val += U_mat[s, k] * S_vec[k] * Vt_mat[k, j]
                if j < n_heads * d:
                    h = j // d
                    dim = j % d
                    Kr[b, s, h, dim] = val
                else:
                    j_v = j - n_heads * d
                    h = j_v // d
                    dim = j_v % d
                    Vr[b, s, h, dim] = val

    mla_out = _attention(Q, Kr, Vr)

    mha_out = _attention(Q, K, V)
    
    gqa_err = 0.0
    mla_err = 0.0
    
    for b in range(batch):
        for s in range(Q.shape[1]):
            for h in range(n_heads):
                for dim in range(d):
                    diff_gqa = math.fabs(gqa_out[b, s, h, dim] - mha_out[b, s, h, dim])
                    if diff_gqa > gqa_err:
                        gqa_err = diff_gqa
                        
                    diff_mla = math.fabs(mla_out[b, s, h, dim] - mha_out[b, s, h, dim])
                    if diff_mla > mla_err:
                        mla_err = diff_mla

    winner = "mla" if mla_err < gqa_err else "gqa"
    return gqa_err, mla_err, winner
