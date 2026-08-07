import numpy as np

def train_step(local_gradients: list[np.ndarray], reduce_fn) -> float:
    global_grad = reduce_fn(local_gradients)
    return float(np.sum(global_grad ** 2))
