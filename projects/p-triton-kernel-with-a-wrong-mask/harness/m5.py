import os
import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))
    import ref

    res = {"matched_sizes_count": 0.0, "all_matched": 0.0}
    try:
        import triton_mask.kernel as k

        sizes = list(range(1, 51))
        matched = 0
        block_sizes = [16, 32, 64, 128]
        for idx, n in enumerate(sizes):
            bs = block_sizes[idx % len(block_sizes)]
            x = np.linspace(-10.0, 10.0, n, dtype=np.float32)
            learner_out = k.process_data(x, n, BLOCK_SIZE=bs)
            ref_out = ref.process_data(x, n, BLOCK_SIZE=bs)
            if np.allclose(learner_out, ref_out):
                matched += 1

        res["matched_sizes_count"] = float(matched)
        if matched == 50:
            res["all_matched"] = 1.0
    except Exception:
        pass
    return res
