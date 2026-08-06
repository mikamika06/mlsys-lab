import sys
from pathlib import Path
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, str(Path(workdir) / "reference"))
    sys.path.insert(0, str(workdir))

    from loraspec.scaling import apply_lora_scaling, compute_scaling_factor

    out = {"scaling_matched": 0.0, "output_delta_matched": 0.0}

    scaling_ok = True
    for alpha in [8.0, 16.0, 32.0]:
        for r in [4, 8, 16, 64]:
            for mode in ["lora", "naive"]:
                want = ref.compute_scaling_factor(alpha, r, mode=mode)
                got = compute_scaling_factor(alpha, r, mode=mode)
                if not np.isclose(want, got):
                    scaling_ok = False
                    out["_note"] = f"Scaling factor mismatch: want {want}, got {got}"
                    break
            if not scaling_ok:
                break
        if not scaling_ok:
            break

    if scaling_ok:
        out["scaling_matched"] = 1.0

    np.random.seed(42)
    delta_ok = True
    for _ in range(5):
        d_in, d_out, r = 128, 256, 16
        x = np.random.randn(4, d_in)
        wa = np.random.randn(r, d_in)
        wb = np.random.randn(d_out, r)
        alpha = 32.0

        for mode in ["lora", "naive"]:
            want = ref.apply_lora_scaling(x, wa, wb, alpha, r, mode=mode)
            got = apply_lora_scaling(x, wa, wb, alpha, r, mode=mode)
            if not np.allclose(want, got, atol=1e-5):
                delta_ok = False
                out["_note"] = "Output delta mismatch"
                break
        if not delta_ok:
            break

    if delta_ok:
        out["output_delta_matched"] = 1.0

    return out
