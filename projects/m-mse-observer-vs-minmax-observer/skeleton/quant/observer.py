import numpy as np


class MinMaxObserver:
    def __init__(self, bits: int = 8, symmetric: bool = True):
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def compute_params(self) -> tuple[float, int]:
        raise NotImplementedError


class MSEObserver:
    def __init__(self, bits: int = 8, symmetric: bool = True, num_bins: int = 100):
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def compute_params(self) -> tuple[float, int]:
        raise NotImplementedError
