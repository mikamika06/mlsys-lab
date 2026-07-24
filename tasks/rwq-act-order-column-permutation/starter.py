import numpy as np


def gptq_act_order(W: np.ndarray, H: np.ndarray, nbits: int, damp: float):
    """
    Compute the act-order column permutation (argsort of diag(H) descending)
    and run column-sequential GPTQ quantization in that order, returning
    (perm, mse) as described in task.md.
    """
    raise NotImplementedError('your code here')
