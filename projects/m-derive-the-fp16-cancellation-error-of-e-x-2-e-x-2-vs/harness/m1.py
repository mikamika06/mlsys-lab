import ref
import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from reduction.variance import one_pass_variance, two_pass_variance
    except ImportError:
        sys.path.pop(0)
        return {"_note": "Could not import reduction.variance"}
    sys.path.pop(0)

    out = {"one_pass_fails": 0.0, "two_pass_succeeds": 0.0}

    x = ref.get_variance_test_data()
    true_var = np.var(x.astype(np.float64))

    try:
        one = float(one_pass_variance(x))
        two = float(two_pass_variance(x))
    except NotImplementedError:
        return out

    err_one = abs(one - true_var) / true_var
    err_two = abs(two - true_var) / true_var

    out["rel_err_one_pass"] = err_one
    out["rel_err_two_pass"] = err_two

    if err_one > 0.5:
        out["one_pass_fails"] = 1.0
    else:
        out["_note"] = f"One pass error was {err_one:.3f}, expected > 0.5 due to float16 cancellation."

    if err_two < 0.1:
        out["two_pass_succeeds"] = 1.0
    else:
        out["_note"] = out.get("_note", "") + f" Two pass error was {err_two:.3f}, expected < 0.1."

    return out
