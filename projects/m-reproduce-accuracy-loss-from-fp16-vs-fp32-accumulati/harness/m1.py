import ref
import numpy as np


def check(workdir):
    from matmul_acc.kernel import compute_matmul_accumulation
    a, b = ref.get_test_inputs()
    out_ref = np.dot(a, b)
    out_learner = compute_matmul_accumulation(a, b, use_fp32_acc=True)
    rel_err = float(np.linalg.norm(out_learner - out_ref) / (np.linalg.norm(out_ref) + 1e-7))
    return {"rel_err": rel_err}
