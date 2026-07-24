import numpy as np


class ValidatedArray:
    """Backing numpy buffer behind a validated `data` property.

    The setter accepts a new value only if it is an np.ndarray with the
    exact shape and dtype declared at construction; otherwise it raises and
    leaves the existing buffer untouched.
    """

    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._data = np.zeros(self.shape, dtype=self.dtype)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"data must be an np.ndarray, got {type(value)!r}")
        if value.shape != self.shape:
            raise ValueError(f"shape mismatch: expected {self.shape}, got {value.shape}")
        if value.dtype != self.dtype:
            raise ValueError(f"dtype mismatch: expected {self.dtype}, got {value.dtype}")
        self._data = value.copy()
