import ref


def check(workdir):
    from occupancy.calc import compute_theoretical_occupancy
    out = {"occupancy_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_occ, want_bind = ref.compute_theoretical_occupancy(cfg)
        try:
            got_occ, got_bind = compute_theoretical_occupancy(cfg)
        except Exception as e:
            out["_note"] = f"config {i} raised {e}"
            return out
        if abs(got_occ - want_occ) < 1e-5 and got_bind == want_bind:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got ({got_occ}, {got_bind}), want ({want_occ}, {want_bind})"
    if ok == len(ref.CONFIGS):
        out["occupancy_matched"] = 1.0
    return out
