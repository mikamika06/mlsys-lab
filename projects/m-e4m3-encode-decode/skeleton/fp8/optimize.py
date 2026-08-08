import numpy as np
from fp8.descale import quantize_and_descale


def find_optimal_scale(
    x: np.ndarray, candidates: list[float]
) -> tuple[float, float]:
    raise NotImplementedError
