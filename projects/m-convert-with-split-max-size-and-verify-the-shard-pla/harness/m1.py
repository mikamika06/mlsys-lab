import ref

def check(workdir):
    from sharder.plan import build_shard_plan

    out = {"plans_matched": 0.0, "total": float(len(ref.MODELS))}
    ok = 0
    for i, cfg in enumerate(ref.MODELS):
        want = ref.plan_shards(cfg)
        got = build_shard_plan(cfg["tensors"], cfg["max_bytes"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, reference {want}"
    out["plans_matched"] = float(ok)
    return out
