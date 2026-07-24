import numpy as np


def fix_shape_range_and_run(lower, upper, x):
    # TODO: incorrectly resets the lower bound whenever the input does not
    # violate the lower guard, causing unnecessary range expansion.
    n = int(x.shape[0])

    new_lower = 0
    new_upper = max(upper, n)

    output = np.asarray(x, dtype=np.float64)
    output = output * 2.0 + 1.0

    return (new_lower, new_upper), output
