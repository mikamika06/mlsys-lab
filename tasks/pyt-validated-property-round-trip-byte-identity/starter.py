import numpy as np


class ValidatedArray:
    """Backing numpy buffer behind a validated `data` property.

    The setter must accept a new value only if it is an np.ndarray with the
    exact shape and dtype declared at construction; otherwise it must raise
    and leave the existing buffer untouched. See task.md.
    """

    def __init__(self, shape: tuple, dtype=np.float32):
        raise NotImplementedError('your code here')
