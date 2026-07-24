from __future__ import annotations

import numpy as np


def kahan_sum(x: np.ndarray) -> float:
    """Kahan (compensated) summation of `x`, via an explicit per-element
    Python loop.

    Parameters
    ----------
    x : np.ndarray
        1-D float64 array.

    Returns
    -------
    float
        The sum of `x`, computed with running-error compensation so the
        result stays accurate even under catastrophic cancellation
        (e.g. a huge value followed by many small ones followed by its
        negation).

    Must be a real, explicit per-element Python `for` loop — no
    `np.sum`, `math.fsum`, or other bulk-summation shortcut. Track the
    low-order bits lost to rounding on each addition in a compensation
    term and fold them back in on the next step.
    """
    raise NotImplementedError('your code here')
