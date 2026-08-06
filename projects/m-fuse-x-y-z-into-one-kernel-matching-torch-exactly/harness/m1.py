import ref
import numpy as np


def check(workdir):
    from triton_ops.fuse import fuse_ops
    cases = ref.get_test_cases()
    max_err = 0.0
    for x, y, z in cases:
        got = fuse_ops(x, y, z)
        want = ref.torch_ref(x, y, z)
        err = np.max(np.abs(got - want) / (np.abs(want) + 1e-7))
        max_err = max(max_err, float(err))
    return {"max_rel_err": max_err}
