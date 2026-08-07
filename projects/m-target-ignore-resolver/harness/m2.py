import ref


def check(workdir):
    from quantres.moe import build_moe_ignore_list

    ok = 0
    out = {"moe_ignores_matched": 0.0}
    for i, struct in enumerate(ref.MOE_STRUCTURES):
        want = ref.build_moe_ignore_list(struct)
        got = build_moe_ignore_list(struct)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"structure {i}: got {got}, reference {want}"
    if ok == len(ref.MOE_STRUCTURES):
        out["moe_ignores_matched"] = 1.0
    return out
