import ref
import numpy as np


def check(workdir):
    from quantlib.scale import compute_scale
    data = ref.generate_test_data()
    want = ref.compute_scale(data)
    try:
        got = compute_scale(data)
    except Exception as e:
        return {"scale_error": 1.0, "_note": f"raised {e}"}
    err = float(abs(want - got) / (abs(want) + 1e-6))
    return {"scale_error": err}
