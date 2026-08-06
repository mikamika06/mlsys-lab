import sys
import os
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import distill.mapping as ref_map
    sys.path.pop(0)

    out = {"mapping_magnitude_match": 0.0}
    try:
        from distill.mapping import evaluate_layer_mapping

        got = evaluate_layer_mapping(ref.STUDENT_STATES, ref.TEACHER_STATES, "uniform")
        want = ref_map.evaluate_layer_mapping(ref.STUDENT_STATES, ref.TEACHER_STATES, "uniform")
        if np.isclose(got, want, atol=1e-4):
            out["mapping_magnitude_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error in m2: {type(e).__name__}: {str(e)[:120]}"
    return out
