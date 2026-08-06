import ref


def check(workdir):
    from ropescaling.scaling import compute_dynamic_ntk_base
    out = {"dynamic_ntk_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_dynamic_ntk_base(cfg["base"], cfg["seq_len"], cfg["max_pos"])
        got = compute_dynamic_ntk_base(cfg["base"], cfg["seq_len"], cfg["max_pos"])
        if abs(got - want) < 1e-4:
            ok += 1
    out["dynamic_ntk_matched"] = float(ok)
    return out
