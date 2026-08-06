import numpy as np
import ref


def check(workdir):
    from loraeval.compare import compute_output_error
    fixtures = ref.get_m1_fixtures()
    max_err = 0.0
    for bw, aa, ab, scaling, x in fixtures:
        got = compute_output_error(bw, aa, ab, scaling, x)
        unfused = np.matmul(x, bw) + scaling * np.matmul(np.matmul(x, aa), ab)
        fused_w = bw + scaling * np.matmul(aa, ab)
        fused = np.matmul(x, fused_w)
        want = float(np.max(np.abs(fused - unfused)))
        err = abs(got - want)
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
