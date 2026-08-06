import ref


def check(workdir):
    from mpsmem.loop import simulate_generation
    out = {"cadence_matched": 0.0, "rel_err": 0.0}
    steps = 20
    cadence = 4
    alloc = 50
    want_h, want_c = ref.simulate_generation(steps, cadence, alloc)
    got_h, got_c = simulate_generation(steps, cadence, alloc)
    if want_c == got_c and want_h == got_h:
        out["cadence_matched"] = 1.0
        out["rel_err"] = 0.0
    else:
        out["rel_err"] = 1.0
    return out
