import numpy as np
import ref


def check(workdir):
    from lora.merge import merge_lora_weights

    out = {"merge_matched": 0.0}
    np.random.seed(42)
    ok = True
    for cfg in ref.CONFIGS:
        w = np.random.randn(cfg["out_features"], cfg["in_features"])
        a = np.random.randn(cfg["rank"], cfg["in_features"])
        b = np.random.randn(cfg["out_features"], cfg["rank"])
        want = ref.merge_weights(w, a, b, cfg["alpha"], cfg["rank"])
        got = merge_lora_weights(w, a, b, cfg["alpha"], cfg["rank"])
        if not np.allclose(got, want, atol=1e-5, rtol=1e-5):
            ok = False
            out["_note"] = f"merge mismatch for config {cfg}"
            break
    if ok:
        out["merge_matched"] = 1.0
    return out
