import ref
import numpy as np


def check(workdir):
    from q4k.analysis import dominant_subblock
    t = ref.get_test_tensor()[:256]
    try:
        got = dominant_subblock(t)
        want = ref.dominant_subblock(t)
        match = 1.0 if got == want else 0.0
        return {"subblock_identified": match}
    except Exception as e:
        return {"subblock_identified": 0.0, "_note": str(e)}
