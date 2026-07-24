import numpy as np


def reconstruct_attention_from_block_logs(block_m: np.ndarray, block_l: np.ndarray,
                                           block_o: np.ndarray) -> np.ndarray:
    """Reconstruct the single global, correctly-normalized attention output
    vector from K blocks' LOCAL online-softmax summaries -- no raw scores
    or values are available, only:

      block_m: (K,)   each block's local max score.
      block_l: (K,)   each block's local softmax denominator
                       sum(exp(score - block_m[k])) over that block only.
      block_o: (K, d) each block's local weighted value accumulator
                       sum(exp(score - block_m[k])[:, None] * value, axis=0)
                       over that block only.

    See task.md for the exact merge rule.
    """
    raise NotImplementedError('your code here')
