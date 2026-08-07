import ref


def check(workdir):
    from tensorgrid.validate import find_optimal_grid, is_feasible

    out = {"validation_matched": 0.0}
    ok = 0
    total = 0
    for gs, bs in ref.TEST_CASES:
        min_l, max_l = ref.reconstruct_length(gs, bs)
        test_lens = [min_l - 1, min_l, (min_l + max_l) // 2, max_l, max_l + 1]
        for tl in test_lens:
            if tl < 0:
                continue
            total += 1
            want_feas = ref.is_feasible(tl, gs, bs)
            try:
                got_feas = is_feasible(tl, gs, bs)
            except Exception:
                got_feas = not want_feas
            if got_feas == want_feas:
                ok += 1

        g_val = gs[0] if isinstance(gs, tuple) else gs
        if g_val > 0:
            total += 1
            mid_tl = (ref.reconstruct_length(gs, bs)[0] + ref.reconstruct_length(gs, bs)[1]) // 2
            want_og = ref.find_optimal_grid(mid_tl, bs)
            try:
                got_og = find_optimal_grid(mid_tl, bs)
            except Exception:
                got_og = -1
            if got_og == want_og:
                ok += 1
    out["validation_matched"] = float(ok)
    out["total"] = float(total)
    return out
