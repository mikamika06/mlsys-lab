import ref


def check(workdir):
    from pipelp.imbalance import find_imbalanced_stage

    out = {"stage_matched": 0.0}
    ok = True
    for logs in ref.LOG_SAMPLES:
        want = ref.find_imbalanced_stage(logs)
        got = find_imbalanced_stage(logs)
        if got != want:
            ok = False
            out["_note"] = f"imbalanced stage mismatch: got {got}, want {want}"
            break
    if ok:
        out["stage_matched"] = 1.0
    return out
