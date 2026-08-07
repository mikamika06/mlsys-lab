import math

def zero_one_adam(params: list[float], grads: list[list[float]], num_ranks: int, lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-08) -> list[float]:
    """ZeRO-1 Adam: partition optimizer states across num_ranks."""
    raise NotImplementedError('your code here')
