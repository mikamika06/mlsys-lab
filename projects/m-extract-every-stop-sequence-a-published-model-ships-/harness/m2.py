import ref
import numpy as np


def check(workdir):
    from extractor.adapter import apply_adapter_and_forward
    out = {"adapter_verified": 0.0}
    w = [[1.0, 2.0], [3.0, 4.0]]
    adapter = ([[0.1, 0.0], [0.0, 0.1]], [[1.0, 0.0], [0.0, 1.0]])
    x = [1.0, 1.0]
    want = ref.apply_adapter_and_forward(w, adapter, x)
    try:
        got = apply_adapter_and_forward(w, adapter, x)
        if np.allclose(want, got):
            out["adapter_verified"] = 1.0
        else:
            out["_note"] = f"output mismatch: got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"execution raised {type(e).__name__}: {str(e)}"
    return out
