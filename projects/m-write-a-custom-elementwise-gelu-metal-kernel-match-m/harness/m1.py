import ref
import numpy as np


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from metalops.gelu import custom_gelu
    x = ref.generate_gelu_inputs()
    want = ref.gelu_reference(x)
    got = custom_gelu(x)
    max_err = float(np.max(np.abs(got - want)))
    return {"max_abs_err": max_err}
