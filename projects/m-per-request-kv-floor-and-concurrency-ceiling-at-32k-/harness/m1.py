import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from kvcapacity.rope import compute_effective_context
    from kvcapacity.floor import per_request_kv_bytes

    out = {"effective_contexts_matched": 0.0, "kv_floors_matched": 0.0}

    ctx_ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_effective_context(cfg)
        got = compute_effective_context(cfg)
        if got == want:
            ctx_ok += 1
        elif "_note" not in out:
            out["_note"] = f"ctx mismatch config {i}: got {got}, want {want}"

    if ctx_ok == len(ref.CONFIGS):
        out["effective_contexts_matched"] = 1.0

    kv_ok = 0
    total_kv_tests = 0
    for cfg in ref.CONFIGS:
        for seq_len in [32768, 131072, 262144]:
            for kv_dt in ["float16", "fp8"]:
                total_kv_tests += 1
                want = ref.per_request_kv_bytes(cfg, seq_len, kv_dt)
                got = per_request_kv_bytes(cfg, seq_len, kv_dt)
                if got == want:
                    kv_ok += 1
                elif "_note" not in out:
                    out["_note"] = f"kv mismatch seq={seq_len} dt={kv_dt}: got {got}, want {want}"

    if kv_ok == total_kv_tests:
        out["kv_floors_matched"] = 1.0

    return out
