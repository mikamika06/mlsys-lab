import ref

def check(workdir):
    from zigzag.assignment import generate_zigzag_assignments

    out = {"assignments_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.generate_zigzag_assignments(cfg["num_tokens"], cfg["world_size"])
        got = generate_zigzag_assignments(cfg["num_tokens"], cfg["world_size"])
        if got == want:
            ok += 1
    out["assignments_matched"] = float(ok)
    return out
