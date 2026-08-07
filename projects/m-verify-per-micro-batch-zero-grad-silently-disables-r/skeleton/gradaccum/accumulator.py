import numpy as np


class SimpleModel:
    def __init__(self, weights: dict[str, np.ndarray]):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def accumulate(self, micro_grad: dict[str, np.ndarray], scale: float = 1.0):
        raise NotImplementedError

    def step(self, lr: float):
        raise NotImplementedError


def run_correct_accumulation(
    model: SimpleModel,
    micro_batch_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
    lr: float,
) -> list[dict[str, np.ndarray]]:
    raise NotImplementedError


def run_buggy_accumulation(
    model: SimpleModel,
    micro_batch_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
    lr: float,
) -> list[dict[str, np.ndarray]]:
    raise NotImplementedError
