import ref


def check(workdir):
    from treeprune.calib import reconstruct_tree_choices as user_func

    out = {"choices_matched": 0.0}
    ok = 0
    for i, fixture in enumerate(ref.FIXTURES):
        want = ref.reconstruct_tree_choices(fixture)
        try:
            got = user_func(fixture)
        except Exception as e:
            out["_note"] = f"fixture {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"fixture {i}: got {got}, reference {want}"
    out["choices_matched"] = float(ok)
    return out
