import numpy as np


def regrow_masks(w, g, mask, k, seed):
    """Two dynamic-sparse-training regrowth strategies, same regrow count.

    mask marks the currently active (nonzero) connections. Both strategies
    pick k currently-inactive positions to reactivate, leaving every
    already-active position untouched:

    - RigL (gradient-informed): reactivate the k inactive positions with
      the largest |g|. Ties broken by lower flat (row-major) index first.
    - SET (random): reactivate k inactive positions chosen uniformly at
      random, without replacement, via
      np.random.default_rng(seed).choice(zero_indices, size=k,
      replace=False) over the flat indices of the inactive positions in
      ascending order.

    Returns (rigl_mask, set_mask): two bool arrays, same shape as mask.
    """
    raise NotImplementedError('your code here')
