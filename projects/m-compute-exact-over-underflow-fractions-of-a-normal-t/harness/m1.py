import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from numdiag.overflow import compute_overflow_underflow_fractions
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Import failed: {type(e).__name__}: {str(e)}"}

    tensors = ref.get_test_tensors()
    dtypes = ["fp16", "bf16"]

    max_rel_err = 0.0

    for name, tensor in tensors.items():
        for dt in dtypes:
            expected = ref.ref_compute_overflow_underflow_fractions(tensor, dt)
            try:
                actual = compute_overflow_underflow_fractions(tensor, dt)
            except Exception as e:
                return {"rel_err": 1.0, "_note": f"Execution failed on {name}/{dt}: {type(e).__name__}: {str(e)}"}

            for k in ["overflow", "underflow"]:
                exp_v = expected[k]
                act_v = actual.get(k, -1.0)
                err = abs(exp_v - act_v) / (abs(exp_v) + 1e-12)
                if err > max_rel_err:
                    max_rel_err = err

    return {"rel_err": float(max_rel_err)}
