import ref


def check(workdir):
    from flopdiag.count import expected_linear_count
    out = {"count_match": 0.0}
    tests = ref.get_m2_tests()
    ok = True
    for cfg, expected in tests:
        got = expected_linear_count(cfg["num_layers"], cfg["hidden_size"], cfg["intermediate_size"])
        if got != expected:
            ok = False
            out["_note"] = f"cfg {cfg}: got {got}, want {expected}"
            break
    if ok:
        out["count_match"] = 1.0
    return out
