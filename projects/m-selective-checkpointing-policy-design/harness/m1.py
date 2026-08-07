import ref

def check(workdir):
    from checkpoint.policy import select_checkpoint_policy
    out = {"policies_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.select_policy(cfg, cfg["budget"])
        got = select_checkpoint_policy(cfg, cfg["budget"])
        if got == want:
            ok += 1
    out["policies_matched"] = float(ok)
    return out
