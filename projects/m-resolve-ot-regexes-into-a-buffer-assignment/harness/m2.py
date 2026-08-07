import ref
from moeoff.vram import compute_vram_saved
from moeoff.optimizer import find_min_cpu_moe

def check(workdir):
    out = {"vram_match": 0.0, "min_moe_match": 0.0}
    saved = compute_vram_saved(ref.TENSORS, ref.PATTERNS, 1, ref.TOTAL_LAYERS)
    if saved == 2000:
        out["vram_match"] = 1.0
    else:
        out["_note"] = f"saved {saved}, want 2000"
        return out

    total_size = sum(s for _, s in ref.TENSORS)
    limit = total_size - 2000
    min_n = find_min_cpu_moe(ref.TENSORS, ref.PATTERNS, ref.TOTAL_LAYERS, limit)
    if min_n == 1:
        out["min_moe_match"] = 1.0
    else:
        out["_note"] = f"min_n {min_n}, want 1"
    return out
