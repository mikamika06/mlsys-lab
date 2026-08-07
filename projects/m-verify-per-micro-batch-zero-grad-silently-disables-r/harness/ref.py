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


def analyze_accumulation_discrepancy(
    correct_grads: list[dict[str, np.ndarray]],
    buggy_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
) -> dict[str, float]:
    if not correct_grads or not buggy_grads or len(correct_grads) != len(buggy_grads):
        return {
            "max_abs_error": 0.0,
            "effective_batch_fraction": 0.0,
            "is_buggy": 0.0,
        }

    max_err = 0.0
    for g_c, g_b in zip(correct_grads, buggy_grads):
        for k in g_c:
            err = np.max(np.abs(g_c[k] - g_b[k]))
            if err > max_err:
                max_err = float(err)

    eff_fraction = 1.0 / float(accum_steps)
    is_buggy = 1.0 if max_err > 1e-5 else 0.0

    return {
        "max_abs_error": float(max_err),
        "effective_batch_fraction": float(eff_fraction),
        "is_buggy": float(is_buggy),
    }


def make_weights(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        "w1": rng.randn(8, 16),
        "b1": rng.randn(8),
        "w2": rng.randn(4, 8),
    }


def make_micro_batch_grads(num_mbs: int = 12, seed: int = 123) -> list[dict[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    grads = []
    for _ in range(num_mbs):
        grads.append({
            "w1": rng.randn(8, 16),
            "b1": rng.randn(8),
            "w2": rng.randn(4, 8),
        })
    return grads
