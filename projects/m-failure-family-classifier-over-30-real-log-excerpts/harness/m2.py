import ref


def check(workdir):
    from faildebug.runbook import get_minimal_fix
    out = {"fixes_match": 0.0}
    ok = True
    for fam in ref.FAMILIES:
        got = get_minimal_fix(fam)
        want = ref.get_minimal_fix(fam)
        if got != want:
            ok = False
            out["_note"] = f"Family {fam}: got {got}, want {want}"
            break
    if ok:
        out["fixes_match"] = 1.0
    return out
