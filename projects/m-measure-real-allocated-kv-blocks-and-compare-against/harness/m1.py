import ref

def check(workdir):
    from kvalloc.blocks import compute_budget
    out = {"budgets_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_budget(cfg["seq_lens"], cfg["block_size"], cfg["num_layers"])
        got = compute_budget(cfg["seq_lens"], cfg["block_size"], cfg["num_layers"])
        if got == want:
            ok += 1
    out["budgets_matched"] = float(ok)
    return out
