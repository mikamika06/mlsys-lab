import numpy as np


def check_fidelity(eager_outputs, ort_outputs, rtol=1e-5, atol=1e-5):
    if len(eager_outputs) != len(ort_outputs):
        return False
    for e, o in zip(eager_outputs, ort_outputs):
        if not np.allclose(e, o, rtol=rtol, atol=atol):
            return False
    return True
