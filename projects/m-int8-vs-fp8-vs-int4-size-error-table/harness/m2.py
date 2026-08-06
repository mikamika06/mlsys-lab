import ref
import numpy as np

def check(workdir):
    from qsim.calibrate import compare_domains

    rs = np.random.RandomState(101)
    act_in = rs.randn(1000, 64).astype(np.float32)
    # create out-of-domain with entirely different scales/variance
    act_out = rs.randn(1000, 64).astype(np.float32) * 8.0

    out = {"calibration_matches": 0.0}
    want_in, want_out = ref.compare_domains(act_in, act_out)

    try:
        got_in, got_out = compare_domains(act_in, act_out)
    except Exception as e:
        out["_note"] = f"Error during compare_domains: {e}"
        return out

    if np.isclose(want_in, got_in) and np.isclose(want_out, got_out):
        out["calibration_matches"] = 1.0
    else:
        out["_note"] = f"Expected errors ({want_in:.4f}, {want_out:.4f}), got ({got_in:.4f}, {got_out:.4f})"

    return out
