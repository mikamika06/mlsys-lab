import ref


def check(workdir):
    from realignment.align import align_tokens

    out = {"alignments_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.align_tokens(cfg["draft_tokens"], cfg["target_tokens"])
        got = align_tokens(cfg["draft_tokens"], cfg["target_tokens"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["alignments_matched"] = float(ok)
    return out
