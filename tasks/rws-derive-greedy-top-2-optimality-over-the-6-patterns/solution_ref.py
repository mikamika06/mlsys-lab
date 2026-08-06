import numpy as np


def greedy_24_prune(W: np.ndarray) -> np.ndarray:
    W_arr = np.asarray(W, dtype=np.float64)
    num_rows = W_arr.shape[0]
    out = []
    for r in range(num_rows):
        row = W_arr[r]
        best = None
        for keep_0 in range(4):
            for keep_1 in range(keep_0 + 1, 4):
                dropped = 0.0
                for i in range(4):
                    if i != keep_0 and i != keep_1:
                        val = row[i]
                        if val < 0.0:
                            dropped += -val
                        else:
                            dropped += val
                if best is None or dropped < best:
                    best = dropped
        out.append(best)
    return np.asarray(out, dtype=np.float64)
