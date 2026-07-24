import itertools
import numpy as np


def fix_transpose_perm(input_tensor, exported_output, torch_reference):
    dims = tuple(range(input_tensor.ndim))
    for perm in itertools.permutations(dims):
        candidate = np.transpose(input_tensor, perm)
        if np.array_equal(candidate, torch_reference):
            return perm
    raise ValueError("no transpose permutation matches reference")
