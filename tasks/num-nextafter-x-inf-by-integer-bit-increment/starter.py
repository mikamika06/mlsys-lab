from __future__ import annotations

import numpy as np


def next_up(x: np.ndarray) -> np.ndarray:
    """Elementwise IEEE-754 nextafter(x, +inf) for finite float32 `x`,
    computed purely via integer manipulation of the raw sign-magnitude
    bit pattern (no `np.nextafter` call).

    Parameters
    ----------
    x : np.ndarray
        Finite float32 array (may contain +0.0, -0.0, subnormals, and
        the largest finite value, but never inf or NaN).

    Returns
    -------
    np.ndarray
        float32 array of the same shape: the next representable float32
        value strictly greater than each element of `x`.

    Hint: float32 bits are `[sign][exponent][mantissa]` — sign-magnitude,
    not two's complement. Moving toward +inf means incrementing the raw
    31-bit magnitude for non-negative x, but DECREMENTING it for negative
    x (since larger magnitude means more negative). Handle the -0.0
    boundary specially.
    """
    raise NotImplementedError('your code here')
