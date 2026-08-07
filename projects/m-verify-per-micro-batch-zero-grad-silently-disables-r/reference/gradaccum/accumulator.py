import numpy as np


class SimpleModel:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = {k: np.array(v, dtype=np.float64, copy=True) for k, v in weights.items()}
        self.grads = {k: np.zeros_like(v, dtype=np.float64) for k, v in weights.items()}

    def zero_grad(self):
        for k in self.grads:
            self.grads[k].fill(0.0)

    def accumulate(self, micro_grad: dict[str, np.ndarray], scale: float = 1.0):
        for k, v in micro_grad.items():
            self.grads[k] += np.array(v, dtype=np.float64) * scale

    def step(self, lr: float):
        for k in self.weights:
            self.weights[k] -= lr * self.grads[k]


def run_correct_accumulation(
    model: SimpleModel,
    micro_batch_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
    lr: float,
) -> list[dict[str, np.ndarray]]:
    recorded_grads = []
    num_mbs = len(micro_batch_grads)

    for i in range(0, num_mbs, accum_steps):
        chunk = micro_batch_grads[i : i + accum_steps]
        model.zero_grad()
        for mb_grad in chunk:
            model.accumulate(mb_grad, scale=1.0 / len(chunk))
        recorded_grads.append({k: v.copy() for k, v in model.grads.items()})
        model.step(lr)

    return recorded_grads


def run_buggy_accumulation(
    model: SimpleModel,
    micro_batch_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
    lr: float,
) -> list[dict[str, np.ndarray]]:
    recorded_grads = []
    num_mbs = len(micro_batch_grads)

    for i in range(0, num_mbs, accum_steps):
        chunk = micro_batch_grads[i : i + accum_steps]
        for mb_grad in chunk:
            model.zero_grad()
            model.accumulate(mb_grad, scale=1.0 / len(chunk))
        recorded_grads.append({k: v.copy() for k, v in model.grads.items()})
        model.step(lr)

    return recorded_grads
