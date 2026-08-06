import math
import numpy as np


def stable_softmax_argmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    rows = x.shape[0]
    cols = x.shape[1]

    out_list = []
    for i in range(rows):
        max_val = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > max_val:
                max_val = x[i, j]

        row_exps = []
        sum_e = 0.0
        for j in range(cols):
            val = math.exp(x[i, j] - max_val)
            row_exps.append(val)
            sum_e += val

        max_prob = -1.0
        best_idx = 0
        for j in range(cols):
            prob = row_exps[j] / sum_e
            if prob > max_prob:
                max_prob = prob
                best_idx = j

        out_list.append(best_idx)

    return np.array(out_list, dtype=np.int64)
