import os
import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))
    import ref

    res = {"tail_indices_match": 0.0, "exact_block_empty": 0.0}
    try:
        import triton_mask.kernel as k

        idx = k.detect_corrupted_indices(100, BLOCK_SIZE=64)
        expected_idx = ref.detect_corrupted_indices(100, BLOCK_SIZE=64)
        if np.array_equal(idx, expected_idx):
            res["tail_indices_match"] = 1.0

        exact_idx = k.detect_corrupted_indices(64, BLOCK_SIZE=64)
        if len(exact_idx) == 0:
            res["exact_block_empty"] = 1.0
    except Exception:
        pass
    return res
