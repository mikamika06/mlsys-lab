import numpy as np


def moe_dispatch_combine(X: np.ndarray, expert_idx: np.ndarray,
                          gate_weight: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Dispatch tokens to their assigned expert, apply that expert's linear
    transform, and combine results back into the original token order,
    scaled by the per-token gate weight.
    """
    X = np.asarray(X, dtype=np.float64)
    expert_idx = np.asarray(expert_idx)
    gate_weight = np.asarray(gate_weight, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, d = X.shape

    order = []
    for i in range(n):
        order.append(i)
    
    for i in range(1, n):
        key = order[i]
        key_val = expert_idx[key]
        j = i - 1
        while j >= 0 and expert_idx[order[j]] > key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    sorted_idx = []
    for i in range(n):
        sorted_idx.append(expert_idx[order[i]])

    X_sorted = []
    for i in range(n):
        X_sorted.append(X[order[i]])

    out_sorted = []
    for i in range(n):
        row = []
        for _ in range(d):
            row.append(0.0)
        out_sorted.append(row)

    start = 0
    while start < n:
        e = int(sorted_idx[start])
        end = start
        while end < n and sorted_idx[end] == e:
            end += 1
        
        for i in range(start, end):
            for j in range(d):
                acc = 0.0
                for k in range(d):
                    acc += X_sorted[i][k] * W[e][k][j]
                out_sorted[i][j] = acc
        start = end

    Y_list = []
    for _ in range(n):
        row = []
        for _ in range(d):
            row.append(0.0)
        Y_list.append(row)

    for i in range(n):
        orig_idx = order[i]
        for j in range(d):
            Y_list[orig_idx][j] = out_sorted[i][j]

    for i in range(n):
        g = gate_weight[i]
        for j in range(d):
            Y_list[i][j] *= g

    Y = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            Y[i, j] = Y_list[i][j]

    return Y
