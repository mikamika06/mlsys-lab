import ref

def check(workdir):
    from chunkplan import planner
    out = {"plans_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.plan_chunks(cfg["prompts"], cfg["budget"])
        got = planner.plan_chunks(cfg["prompts"], cfg["budget"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["plans_matched"] = float(ok)
    return out
