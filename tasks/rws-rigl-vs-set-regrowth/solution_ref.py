import numpy as np


def regrow_masks(w, g, mask, k, seed):
    """Two dynamic-sparse-training regrowth strategies, same regrow count.

    `mask` marks the currently active (nonzero) connections of a sparse
    layer. Both strategies pick `k` currently-INACTIVE positions to
    reactivate, leaving every already-active position untouched:

    - **RigL** (gradient-informed): reactivate the `k` inactive positions
      with the largest gradient magnitude `|g|`. Ties are broken by lower
      flat (row-major) index first.
    - **SET** (random): reactivate `k` inactive positions chosen uniformly
      at random, without replacement, using
      `np.random.default_rng(seed).choice(zero_indices, size=k, replace=False)`
      over the flat (row-major) indices of the inactive positions, in
      ascending order.

    Parameters
    ----------
    w : np.ndarray
        Dense weight array (unused by the regrowth decision itself, present
        only because a real DST step also needs it to initialize the newly
        grown connections -- kept in the signature for realism).
    g : np.ndarray, same shape as w
        Dense gradient array.
    mask : np.ndarray[bool], same shape as w
        Current active-connection mask.
    k : int
        Number of inactive positions to reactivate (0 <= k <= number of
        inactive positions).
    seed : int
        RNG seed for the SET strategy.

    Returns
    -------
    rigl_mask : np.ndarray[bool], same shape as mask
    set_mask : np.ndarray[bool], same shape as mask
    """
    mask = np.asarray(mask, dtype=bool)
    g = np.asarray(g, dtype=np.float64)
    shape = mask.shape

    flat_mask = mask.ravel()
    flat_abs_g = np.abs(g).ravel()
    zero_idx = np.flatnonzero(~flat_mask)

    # RigL: top-k |g| among the inactive positions.
    order = np.argsort(-flat_abs_g[zero_idx], kind="stable")
    rigl_pick = zero_idx[order[:k]]
    rigl_mask = flat_mask.copy()
    rigl_mask[rigl_pick] = True

    # SET: k inactive positions chosen uniformly at random.
    rng = np.random.default_rng(seed)
    set_pick = rng.choice(zero_idx, size=k, replace=False)
    set_mask = flat_mask.copy()
    set_mask[set_pick] = True

    return rigl_mask.reshape(shape), set_mask.reshape(shape)
