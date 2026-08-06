import ref


def check(workdir):
    from benchkit import decay, parse

    rows = parse.load_all(ref.files())
    out = {"table_match": 1.0, "separability_match": 1.0, "slope_sign": 0.0,
           "catches_speedup": 0.0}
    for model in ref.models():
        want = ref.expect_decay(ref.raw(), model)
        got = decay.decay_table(rows, model=model)
        if len(got) != len(want):
            out["table_match"] = 0.0
            continue
        for g, w in zip(got, want):
            if g.get("depth") != w["depth"]:
                out["table_match"] = 0.0
            if not ref.near(g.get("tokens_per_second", -1), w["tokens_per_second"], 1e-9):
                out["table_match"] = 0.0
            if not ref.near(g.get("loss_fraction", -99), w["loss_fraction"], 1e-9):
                out["table_match"] = 0.0
            if g.get("separable_from_empty") != w["separable_from_empty"]:
                out["separability_match"] = 0.0
        if any(w["loss_fraction"] < 0 for w in want):
            if any(g.get("loss_fraction", 0) < 0 for g in got):
                out["catches_speedup"] = 1.0
    big = max(ref.models(), key=lambda m: len(ref.expect_decay(ref.raw(), m)))
    table = decay.decay_table(rows, model=big)
    if table and decay.slope_per_1k(table) < 0:
        out["slope_sign"] = 1.0
    return out
