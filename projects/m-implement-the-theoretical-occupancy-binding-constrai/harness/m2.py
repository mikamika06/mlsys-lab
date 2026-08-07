import ref


def check(workdir):
    from occupancy.calc import find_optimal_register_cap
    out = {"cap_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_cap = ref.find_optimal_register_cap(cfg, cfg["spill_limit"])
        try:
            got_cap = find_optimal_register_cap(cfg, cfg["spill_limit"])
        except Exception as e:
            out["_note"] = f"config {i} raised {e}"
            return out
        if got_cap == want_cap:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got cap {got_cap}, want {want_cap}"
    if ok == len(ref.CONFIGS):
        out["cap_matched"] = 1.0
    return out
