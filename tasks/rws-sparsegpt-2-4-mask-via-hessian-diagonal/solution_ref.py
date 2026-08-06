import numpy as np


def sparsegpt_24_prune(W: np.ndarray, X: np.ndarray, damp: float = 0.01):
    """SparseGPT-style 2:4 structured pruning: Hessian-diagonal mask
    selection + one-shot OBS/OBC compensation update.

    See task.md for the exact formulas. Returns (mask, W_hat).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    O, I = W.shape

    H = np.zeros((I, I), dtype=np.float64)
    n_rows_x = X.shape[0]
    for i in range(I):
        for j in range(I):
            acc = 0.0
            for k in range(n_rows_x):
                acc += X[k, i] * X[k, j]
            H[i, j] = acc

    diag_sum = 0.0
    for i in range(I):
        diag_sum += H[i, i]
    mean_diag = diag_sum / I

    for i in range(I):
        H[i, i] += damp * mean_diag

    Hinv = np.zeros((I, I), dtype=np.float64)
    aug = np.zeros((I, 2 * I), dtype=np.float64)
    for i in range(I):
        for j in range(I):
            aug[i, j] = H[i, j]
        aug[i, I + i] = 1.0

    for i in range(I):
        pivot = aug[i, i]
        for j in range(2 * I):
            aug[i, j] /= pivot
        for r in range(I):
            if r != i:
                factor = aug[r, i]
                for j in range(2 * I):
                    aug[r, j] -= factor * aug[i, j]

    for i in range(I):
        for j in range(I):
            Hinv[i, j] = aug[i, I + j]

    diag_hinv = np.zeros(I, dtype=np.float64)
    for i in range(I):
        diag_hinv[i] = Hinv[i, i]

    mask = np.ones((O, I), dtype=np.int64)
    for o in range(O):
        for g0 in range(0, I, 4):
            scores = []
            for k in range(4):
                idx_val = g0 + k
                sc = (W[o, idx_val] ** 2) / diag_hinv[idx_val]
                scores.append((sc, k))

            for i_sort in range(len(scores)):
                for j_sort in range(len(scores) - 1 - i_sort):
                    if scores[j_sort][0] > scores[j_sort + 1][0]:
                        scores[j_sort], scores[j_sort + 1] = scores[j_sort + 1], scores[j_sort]

            for rank in range(2):
                orig_k = scores[rank][1]
                mask[o, g0 + orig_k] = 0

    W_hat = np.zeros((O, I), dtype=np.float64)
    for o in range(O):
        for i in range(I):
            W_hat[o, i] = W[o, i]

    for o in range(O):
        S = []
        for i in range(I):
            if mask[o, i] == 0:
                S.append(i)
        
        len_s = len(S)
        if len_s == 0:
            continue

        w_S = np.zeros(len_s, dtype=np.float64)
        for idx_s, orig_i in enumerate(S):
            w_S[idx_s] = W[o, orig_i]

        Hinv_SS = np.zeros((len_s, len_s), dtype=np.float64)
        for r in range(len_s):
            for c in range(len_s):
                Hinv_SS[r, c] = Hinv[S[r], S[c]]

        aug_sys = np.zeros((len_s, len_s + 1), dtype=np.float64)
        for r in range(len_s):
            for c in range(len_s):
                aug_sys[r, c] = Hinv_SS[r, c]
            aug_sys[r, len_s] = w_S[r]

        for r in range(len_s):
            pivot = aug_sys[r, r]
            for c in range(len_s + 1):
                aug_sys[r, c] /= pivot
            for other_r in range(len_s):
                if other_r != r:
                    factor = aug_sys[other_r, r]
                    for c in range(len_s + 1):
                        aug_sys[other_r, c] -= factor * aug_sys[r, c]

        sol_vec = np.zeros(len_s, dtype=np.float64)
        for r in range(len_s):
            sol_vec[r] = aug_sys[r, len_s]

        delta = np.zeros(I, dtype=np.float64)
        for i in range(I):
            acc_val = 0.0
            for r in range(len_s):
                acc_val += Hinv[i, S[r]] * sol_vec[r]
            delta[i] = -acc_val

        for i in range(I):
            W_hat[o, i] = W[o, i] + delta[i]

        for orig_i in S:
            W_hat[o, orig_i] = 0.0

    return mask, W_hat
