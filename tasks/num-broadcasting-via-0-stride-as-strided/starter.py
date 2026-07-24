import numpy as np
from numpy.lib.stride_tricks import as_strided


def broadcast_to_strided(a, shape):
    """Emulate ``np.broadcast_to(a, shape)`` with an ``as_strided`` 0-stride view.

    Returns a read-only view that shares memory with ``a``; raises ``ValueError``
    if ``a.shape`` cannot be broadcast to ``shape``.
    """
    raise NotImplementedError('your code here')
