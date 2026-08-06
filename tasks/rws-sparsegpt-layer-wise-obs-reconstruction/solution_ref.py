import math
import numpy as np


def sparsegpt_layerwise(W, X, sparsity, lam):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, d = W_work.shape
    n_cols = X.shape[1]

    X_XT = [
        [
            sum(X[i, k] * X[j, k] for k in range(n_cols))
            for j in range(d)
        ]
        for i in range(d)
    ]

    H = [
        [2.0 * X_XT[i][j] + (lam if i == j else 0.0) for j in range(d)]
        for i in range(d)
    ]

    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = H[i][i] - s
                if val < 0.0:
                    val = 0.0
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0.0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (H[i][j] - s) / L[j][j]

    invL = [[0.0] * d for _ in range(d)]
    for i in range(d):
        invL[i][i] = 1.0 / L[i][i]
        for j in range(i):
            s = sum(L[i][k] * invL[k][j] for k in range(j, i))
            invL[i][j] = -s / L[i][i]

    Hinv = [
        [
            sum(invL[k][r] * invL[k][c] for k in range(d))
            for c in range(d)
        ]
        for r in range(d)
    ]

    remove = int(m * d * sparsity)
    diag_Hinv = [Hinv[j][j] for j in range(d)]

    scores_list = []
    for i in range(m):
        for j in range(d):
            val = W_work[i, j]
            score = (val * val) / diag_Hinv[j]
            scores_list.append((score, i * d + j))

    sorted_scores = sorted(scores_list, key=lambda x: x[0])
    remove_idx = [item[1] for item in sorted_scores[:remove]]

    mask = np.ones_like(W_work, dtype=bool)

    for idx in remove_idx:
        i, j = divmod(int(idx), d)
        if not mask[i, j]:
            continue
        value = W_work[i, j]
        mask[i, j] = False
        W_work[i, j] = 0.0
        denom = Hinv[j][j]
        for c in range(d):
            W_work[i, c] += (-value * Hinv[j][c]) / denom
        W_work[i, j] = 0.0

    return W_work, mask
