import ref
import numpy as np

def check(workdir):
    try:
        from compression.pipeline import measure_gains
    except ImportError:
        return {"m4_ok": 0.0}

    w = ref.generate_fixture()
    w_comp = ref.joint_recipe(w, 0.25, 8)
    try:
        res = measure_gains(w, w_comp, 8)
    except Exception:
        return {"m4_ok": 0.0}

    nz = np.count_nonzero(w_comp)
    expected_bits = nz * 8
    expected_speedup = w.size / nz

    if res.get("size_bits") == expected_bits and abs(res.get("speedup_factor", 0) - expected_speedup) < 1e-5:
        return {"m4_ok": 1.0}
    return {"m4_ok": 0.0}
