import ref
import numpy as np

def check(workdir):
    from bnb_sim.matmul import mixed_precision_matmul
    cases = ref.generate_cases()
    errors = []
    np.random.seed(100)
    for case in cases:
        A = np.random.randn(16, case.shape[0]).astype(np.float32)
        want_res = ref.mixed_precision_matmul(A, case)
        got_res = mixed_precision_matmul(A, case)
        err = np.linalg.norm(got_res - want_res) / (np.linalg.norm(want_res) + 1e-8)
        errors.append(err)
    max_err = float(np.max(errors))
    out = {"rel_err": max_err}
    return out
