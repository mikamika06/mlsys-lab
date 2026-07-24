import numpy as np


def fix_transpose_perm(input_tensor, exported_output, torch_reference):
    # TODO: recover the graph export transpose permutation instead of assuming
    # that the exported operator already has the identity layout.
    return tuple(range(input_tensor.ndim))
