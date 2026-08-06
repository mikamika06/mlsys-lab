import sys
import os
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import distill.metrics as ref_m
    sys.path.pop(0)

    out = {"losses_matched": 0.0}
    try:
        from distill.metrics import compute_hidden_state_mse, compute_cosine_loss

        s_mse = compute_hidden_state_mse(ref.STUDENT_STATES, ref.TEACHER_STATES[:5])
        s_cos = compute_cosine_loss(ref.STUDENT_STATES, ref.TEACHER_STATES[:5])

        want_mse = ref_m.compute_hidden_state_mse(ref.STUDENT_STATES, ref.TEACHER_STATES[:5])
        want_cos = ref_m.compute_cosine_loss(ref.STUDENT_STATES, ref.TEACHER_STATES[:5])

        matches = 0
        if np.isclose(s_mse, want_mse, atol=1e-4):
            matches += 1
        if np.isclose(s_cos, want_cos, atol=1e-4):
            matches += 1
        out["losses_matched"] = float(matches)
    except Exception as e:
        out["_note"] = f"error in m1: {type(e).__name__}: {str(e)[:120]}"
    return out
