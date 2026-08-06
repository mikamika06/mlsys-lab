import ref


def check(workdir):
    from kvcalc import memory

    out = {"memory_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_kv_bytes(cfg, 4096, 1)
        got = memory.compute_kv_bytes(cfg, 4096, 1)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i} ({cfg['name']}): got {got}, reference {want}"
    out["memory_matched"] = float(ok)
    return out
