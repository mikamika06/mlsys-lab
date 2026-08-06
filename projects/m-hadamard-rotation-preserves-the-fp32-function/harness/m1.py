import ref
import numpy as np

def check(workdir):
    from hadamard.rotation import rotate_activation, rotate_weight

    x = ref.TEST_X
    w = ref.TEST_W

    x_rot = rotate_activation(x, ref.H)
    w_rot = rotate_weight(w, ref.H)

    got = np.matmul(x_rot, w_rot)
    want = ref.baseline_output(x, w)

    diff = np.max(np.abs(got - want))
    out = {"max_abs_err": float(diff)}
    return out
