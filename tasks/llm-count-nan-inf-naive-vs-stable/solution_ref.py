import math
import numpy as np

def count_nonfinite_in_naive_softmax(x: np.ndarray) -> int:
    """
    Compute the naïve softmax of each row in `x` and count how many entries
    are not finite (NaN or Inf).
    """
    count = 0
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    for i in range(num_rows):
        row_exps = []
        sum_exp = 0.0
        for j in range(num_cols):
            val = float(x[i, j])
            try:
                e = math.exp(val)
            except OverflowError:
                e = float('inf')
            row_exps.append(e)
            sum_exp += e

        for e in row_exps:
            if sum_exp == 0.0:
                count += 1
            else:
                sm = e / sum_exp
                if not math.isfinite(sm):
                    count += 1

    return int(count)
