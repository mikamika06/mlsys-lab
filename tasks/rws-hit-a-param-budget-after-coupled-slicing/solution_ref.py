import numpy as np


def _total_params(coefs: np.ndarray, consts: np.ndarray, d: int) -> int:
    dim0 = coefs[:, 0] * d + consts[:, 0]
    dim1 = coefs[:, 1] * d + consts[:, 1]
    return int(np.sum(dim0 * dim1))


def pick_width_for_budget(coefs: np.ndarray, consts: np.ndarray, widths: np.ndarray, budget: int) -> tuple[int, int]:
    """Sweep every candidate width, compute the exact coupled parameter
    count P(d) = sum_t shape_t(d)[0] * shape_t(d)[1], and return the
    largest width whose P(d) fits `budget` (falling back to the width with
    the smallest P(d) if none fits), together with its exact param count.
    """
    coefs = np.asarray(coefs, dtype=np.int64)
    consts = np.asarray(consts, dtype=np.int64)
    widths = np.asarray(widths, dtype=np.int64)
    budget = int(budget)

    counts = np.array([_total_params(coefs, consts, int(d)) for d in widths])
    feasible = widths[counts <= budget]
    if feasible.size > 0:
        chosen = int(feasible.max())
    else:
        chosen = int(widths[int(np.argmin(counts))])

    return chosen, _total_params(coefs, consts, chosen)
