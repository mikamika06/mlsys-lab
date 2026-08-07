import numpy as np


def recover_block_size(trace):
    ptrs = [op["ptr"] for op in trace if "ptr" in op]
    if not ptrs:
        return 0
    diffs = np.diff(ptrs)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return len(ptrs)
    step = int(np.gcd.reduce(np.abs(nonzero)))
    if step == 0:
        return int(len(ptrs))
    return int(np.abs(diffs[0]) // step) if step != 0 else len(ptrs)


def recover_mask_shape(trace):
    masks = [op["mask"] for op in trace if "mask" in op]
    if not masks:
        return (0,)
    m = np.array(masks[0])
    return m.shape
