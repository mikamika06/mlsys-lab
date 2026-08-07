import sys
import numpy as np

sys.path.insert(0, ".")
from sink_ablate.ablation import drop_token_0

def test_drop_token_0_no_inplace_mutation():
    original = np.random.rand(2, 4, 10, 10)
    original = np.tril(original)
    row_sums = original.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    original /= row_sums

    copy_orig = original.copy()
    _ = drop_token_0(original)

    assert np.array_equal(original, copy_orig), "drop_token_0 modified the input array in-place"
