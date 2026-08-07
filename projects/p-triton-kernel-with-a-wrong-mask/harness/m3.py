import os
import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))
    import ref

    res = {"mask_correct": 0.0, "masked_process_ok": 0.0}
    try:
        import triton_mask.kernel as k

        offsets = np.arange(128, dtype=np.int32)
        mask = k.compute_block_mask(offsets, 100)
        expected_mask = ref.compute_block_mask(offsets, 100)
        if np.array_equal(mask, expected_mask):
            res["mask_correct"] = 1.0

        x = np.arange(73, dtype=np.float32) + 1.0
        out = k.process_data(x, 73, BLOCK_SIZE=32)
        expected_out = ref.process_data(x, 73, BLOCK_SIZE=32)
        if np.allclose(out, expected_out):
            res["masked_process_ok"] = 1.0
    except Exception:
        pass
    return res
