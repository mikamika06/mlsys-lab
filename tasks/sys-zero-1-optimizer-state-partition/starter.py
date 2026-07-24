import numpy as np

def zero_one_adam(params, grads, num_ranks, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """ZeRO-1 Adam: partition optimizer states across num_ranks."""
    raise NotImplementedError("Implement ZeRO-1 partitioned Adam")
