def is_2x4_sparse(tensor):
    import numpy as np
    arr = np.asarray(tensor)
    last = arr.shape[-1]
    if last % 4 != 0:
        return False
    groups = arr.reshape(*arr.shape[:-1], -1, 4)
    nonzeros = np.count_nonzero(groups, axis=-1)
    return bool(np.all(nonzeros == 2))
