import ref
import numpy as np

def check(workdir):
    from autotp.model import estimate_decode_latency
    out = {"latency_matched": 0.0, "total_configs": float(len(ref.CONFIGS) * len(ref.HW_PROFILES) * 2)}
    ok = 0
    for cfg in ref.CONFIGS:
        for hw in ref.HW_PROFILES:
            for bs in [1, 16]:
                want = ref.estimate_decode_latency(cfg, hw, 2, bs)
                got = estimate_decode_latency(cfg, hw, 2, bs)
                if abs(want - got) / (want + 1e-9) < 1e-3:
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"mismatch for config, got {got}, want {want}"
    out["latency_matched"] = float(ok)
    return out
