import numpy as np


def compound_error_bound(W: np.ndarray, X: np.ndarray, sparsity: float, nbits: int):
    """
    Compute (e_prune, e_quant, e_compound) as described in task.md:
      - prune W (zero the lowest-magnitude `sparsity` fraction, globally),
      - quantize the pruned weights (per-row symmetric RTN, `nbits` bits),
      - report the three relative output errors (through X) of prune-only,
        quant-only-on-the-pruned-weights, and the full compound transform.
    """
    raise NotImplementedError('your code here')
