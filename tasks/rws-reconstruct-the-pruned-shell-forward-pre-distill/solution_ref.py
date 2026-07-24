import numpy as np


def pruned_shell_forward(W, b, x, keep_rows, keep_cols):
    Wp = np.asarray(W, dtype=np.float64)[keep_rows][:, keep_cols]
    bp = np.asarray(b, dtype=np.float64)[keep_rows]
    xp = np.asarray(x, dtype=np.float64)[keep_cols]
    return Wp @ xp + bp
