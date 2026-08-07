import numpy as np


def per_example_loop(fn, x_batch, axis=0):
    """Applies fn to each single example slice along axis and stacks results."""
    slices = np.swapaxes(x_batch, 0, axis)
    outs = [fn(slices[i]) for i in range(slices.shape[0])]
    res = np.stack(outs, axis=0)
    if axis != 0:
        res = np.swapaxes(res, 0, axis)
    return res


def verify_vmap_matches(fn_single, fn_batched, x_batch, axis=0):
    """Compares per_example_loop(fn_single) against fn_batched and returns max absolute error."""
    ref_out = per_example_loop(fn_single, x_batch, axis=axis)
    vmap_out = fn_batched(x_batch)
    return float(np.max(np.abs(ref_out - vmap_out)))
