import ref


def check(workdir):
    from moediag.params import count_parameters

    out = {"params_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.count_parameters_ref(cfg)
        got = count_parameters(cfg)
        if got != want:
            ok = False
            out["_note"] = f"config {cfg} wanted {want}, got {got}"
            break
    if ok:
        out["params_matched"] = 1.0
    return out
