import ref

def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import quant_recipes.allocator as alloc
    out = {"greedy_matches": 0.0, "counterexample_valid": 0.0}

    ok_g = 0
    total = float(len(ref.PROFILES))
    for p, ex, bdg in ref.PROFILES:
        try:
            want_g = ref.greedy_alloc(p, ex, bdg)
            got_g = alloc.greedy_alloc(p, ex, bdg)
            if want_g == got_g:
                ok_g += 1
        except Exception:
            pass
    out["greedy_matches"] = ok_g / total

    try:
        p, ex, bdg = alloc.find_greedy_counterexample()
        want_opt = ref.optimal_alloc(p, ex, bdg)
        want_grd = ref.greedy_alloc(p, ex, bdg)
        b_opt, s_opt = ref.eval_alloc(p, ex, want_opt)
        b_grd, s_grd = ref.eval_alloc(p, ex, want_grd)

        if b_opt <= bdg and b_grd <= bdg and s_opt < s_grd - 1e-5:
            out["counterexample_valid"] = 1.0
    except Exception as e:
        out["_note"] = str(e)

    return out
