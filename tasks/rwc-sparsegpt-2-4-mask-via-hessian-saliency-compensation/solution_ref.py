import numpy as np


def sparsegpt_2_4(W: np.ndarray, X: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    s = X.shape[0]
    n_rows_x = X.shape[0]
    n_cols_x = X.shape[1]
    
    H = np.zeros((n_cols_x, n_cols_x), dtype=np.float64)
    for i in range(n_cols_x):
        for j in range(n_cols_x):
            acc = 0.0
            for k in range(n_rows_x):
                acc += X[k, i] * X[k, j]
            H[i, j] = acc / s
            if i == j:
                H[i, j] += 1e-4

    Hinv = np.zeros_like(H, dtype=np.float64)
    A = H.copy()
    I_mat = np.zeros((n_cols_x, n_cols_x), dtype=np.float64)
    for i in range(n_cols_x):
        I_mat[i, i] = 1.0
    Hinv = I_mat.copy()

    for i in range(n_cols_x):
        pivot = A[i, i]
        for j in range(n_cols_x):
            A[i, j] /= pivot
            Hinv[i, j] /= pivot
        for r in range(n_cols_x):
            if r != i:
                factor = A[r, i]
                for c in range(n_cols_x):
                    A[r, c] -= factor * A[i, c]
                    Hinv[r, c] -= factor * Hinv[i, c]

    mask = np.zeros(W.shape, dtype=np.int64)
    W_hat = W.copy()

    n_rows_w = W.shape[0]
    n_cols_w = W.shape[1]

    for r in range(n_rows_w):
        for start in range(0, n_cols_w, 4):
            cols = []
            for offset in range(4):
                cols.append(start + offset)
            
            scores = []
            for c in cols:
                sc = (W[r, c] ** 2) / Hinv[c, c]
                scores.append(sc)
            
            paired = []
            for idx in range(len(cols)):
                paired.append((cols[idx], scores[idx]))
            
            for i_idx in range(len(paired)):
                for j_idx in range(i_idx + 1, len(paired)):
                    if paired[j_idx][1] < paired[i_idx][1] or (paired[j_idx][1] == paired[i_idx][1] and paired[j_idx][0] < paired[i_idx][0]):
                        temp = paired[i_idx]
                        paired[i_idx] = paired[j_idx]
                        paired[j_idx] = temp
            
            ranked = []
            for p in paired:
                ranked.append(p[0])
            
            for i_idx in range(2, len(ranked)):
                c = ranked[i_idx]
                mask[r, c] = 1
            
            pruned = []
            for c in cols:
                if mask[r, c] == 0:
                    pruned.append(c)
            
            for c in pruned:
                old = W_hat[r, c]
                for k in cols:
                    if mask[r, k] == 1:
                        W_hat[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                W_hat[r, c] = 0.0

    return mask, W_hat
