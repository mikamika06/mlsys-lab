import numpy as np


def kahan_sum(x: np.ndarray) -> float:
    """Classic Kahan compensated summation (unmodified — keeps its blind spot).

    c = (t - s) - y implicitly assumes |s| >= |x| on every step; when a much
    larger addend arrives while the running sum is still small, the
    compensation recovers nothing and low-order bits of `s` are lost for good.
    """
    raise NotImplementedError("your code here")


def neumaier_sum(x: np.ndarray) -> float:
    """Kahan-Neumaier (Kahan-Babuska) summation: magnitude-checked compensation.

    Picks which of the running sum `s` and the new term `xi` is larger before
    computing the lost bits, so the compensation stays exact regardless of
    which operand dominates.
    """
    raise NotImplementedError("your code here")
