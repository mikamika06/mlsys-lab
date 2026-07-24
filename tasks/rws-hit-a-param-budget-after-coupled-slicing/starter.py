import numpy as np


def pick_width_for_budget(coefs: np.ndarray, consts: np.ndarray, widths: np.ndarray, budget: int) -> tuple[int, int]:
    """Sweep every candidate width, compute the exact coupled parameter
    count P(d) = sum_t shape_t(d)[0] * shape_t(d)[1], and return the
    largest width whose P(d) fits `budget` (falling back to the width with
    the smallest P(d) if none fits), together with its exact param count.

    Returns (chosen_width, param_count) as plain Python ints.
    """
    raise NotImplementedError('your code here')
