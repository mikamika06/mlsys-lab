import ref
import numpy as np


def check(workdir):
    from gradclip.quant import simulate_nf4_cycles
    out = {"error_matched": 0.0}
    tensor = np.linspace(-0.8, 0.8, 50, dtype=np.float32)
    try:
        _, got_err = simulate_nf4_cycles(tensor, 3)
        _, want_err = ref.simulate_nf4_cycles(tensor, 3)
        if np.isclose(got_err, want_err, atol=1e-5):
            out["error_matched"] = 1.0
        else:
            out["_note"] = f"error mismatch: got {got_err}, want {want_err}"
    except Exception as e:
        out["_note"] = f"error during execution: {str(e)}"
    return out
