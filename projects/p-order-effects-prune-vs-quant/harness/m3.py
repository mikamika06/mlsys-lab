import ref
import numpy as np

def check(workdir):
    try:
        from compression.pipeline import joint_recipe
    except ImportError:
        return {"m3_ok": 0.0}

    w = ref.generate_fixture()
    try:
        out = joint_recipe(w, 0.5, 4)
    except Exception:
        return {"m3_ok": 0.0}

    ref_out = ref.joint_recipe(w, 0.5, 4)
    if np.allclose(out, ref_out, atol=1e-5):
        return {"m3_ok": 1.0}
    return {"m3_ok": 0.0}
