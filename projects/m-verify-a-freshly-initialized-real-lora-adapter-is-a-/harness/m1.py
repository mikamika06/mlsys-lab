import ref
import numpy as np

def check(workdir):
    from loraadapter.adapter import LoRALinear
    out = {"shapes_and_init_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        layer = LoRALinear(cfg["in_features"], cfg["out_features"], rank=cfg["rank"], alpha=cfg["alpha"])
        b_is_zero = np.allclose(layer.lora_b, 0.0, atol=1e-12)
        shapes_ok = (layer.lora_a.shape == (cfg["rank"], cfg["in_features"]) and
                     layer.lora_b.shape == (cfg["out_features"], cfg["rank"]) and
                     layer.weight.shape == (cfg["out_features"], cfg["in_features"]))
        if b_is_zero and shapes_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: b_is_zero={b_is_zero}, shapes_ok={shapes_ok}"
    out["shapes_and_init_matched"] = float(ok)
    return out
