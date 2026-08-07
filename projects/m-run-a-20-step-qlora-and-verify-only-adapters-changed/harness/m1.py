import ref
import numpy as np


def check(workdir):
    out = {"forward_match": 0.0}
    try:
        from qlora.layer import LinearQLoRA
    except ImportError:
        out["_note"] = "Could not import qlora.layer.LinearQLoRA"
        return out

    rng = np.random.default_rng(1337)
    X = rng.normal(0, 1, size=(16, 32)).astype(np.float32)

    ref_layer = ref.LinearQLoRA(32, 64, seed=42)
    try:
        student_layer = LinearQLoRA(32, 64, seed=42)
        want = ref_layer.forward(X)
        got = student_layer.forward(X)
    except Exception as e:
        out["_note"] = f"Failed during forward pass: {e}"
        return out

    if np.allclose(want, got, atol=1e-5):
        out["forward_match"] = 1.0
    else:
        out["_note"] = "Forward output mismatch with reference implementation"
    return out
