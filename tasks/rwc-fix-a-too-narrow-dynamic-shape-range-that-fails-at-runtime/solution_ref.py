import numpy as np


def fix_shape_range_and_run(lower, upper, x):
    n = int(x.shape[0])

    new_lower = lower
    new_upper = upper

    if n < new_lower:
        new_lower = n
    if n > new_upper:
        new_upper = n

    output = np.asarray(x, dtype=np.float64)
    output = output * 2.0 + 1.0

    return (new_lower, new_upper), output
