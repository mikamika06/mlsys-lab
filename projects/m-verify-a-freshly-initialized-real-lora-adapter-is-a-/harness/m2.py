import ref
import numpy as np

def check(workdir):
    from loraadapter.adapter import LoRALinear
    from loraadapter.verify import verify_no_op

    rng = np.random.default_rng(123)
    ok = True
    max_err = 0.0
    for cfg in ref.CONFIGS:
        layer = LoRALinear(cfg["in_features"], cfg["out_features"], rank=cfg["rank"], alpha=cfg["alpha"])
        x = rng.normal(0, 1, (16, cfg["in_features"]))
        is_noop, err = verify_no_op(layer, x, tol=1e-7)
        max_err = max(max_err, err)
        if not is_noop:
            ok = False
            break

    out = {"max_abs_err_satisfied": 1.0 if ok else 0.0}
    if not ok:
        out["_note"] = f"max observed error {max_err} exceeded tolerance"
    return out
