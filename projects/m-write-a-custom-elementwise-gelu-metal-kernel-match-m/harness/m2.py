import ref
import numpy as np


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from metalops.matmul import custom_matmul
    a, b = ref.generate_matmul_inputs()
    want = ref.matmul_reference(a, b)
    got = custom_matmul(a, b)
    max_err = float(np.max(np.abs(got - want)))
    return {"max_abs_err": max_err}
