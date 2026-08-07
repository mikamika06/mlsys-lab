import ref
import math

def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import quant_recipes.allocator as alloc
    out = {"uniform_matches": 0.0, "optimal_matches": 0.0, "optimal_valid": 0.0}

    ok_u, ok_o, ok_v = 0, 0, 0
    total = float(len(ref.PROFILES))

    for p, ex, bdg in ref.PROFILES:
        try:
            want_u = ref.uniform_alloc(p, ex, bdg)
            got_u = alloc.uniform_alloc(p, ex, bdg)
            if want_u == got_u:
                ok_u += 1
        except Exception:
            pass

        try:
            want_o = ref.optimal_alloc(p, ex, bdg)
            got_o = alloc.optimal_alloc(p, ex, bdg)

            try:
                b_got, s_got = ref.eval_alloc(p, ex, got_o)
            except KeyError:
                b_got, s_got = float('inf'), float('inf')

            b_want, s_want = ref.eval_alloc(p, ex, want_o)

            if b_got <= bdg and all(got_o.get(n) == 16 for n in ex):
                ok_v += 1

            if math.isclose(s_got, s_want, rel_tol=1e-5):
                ok_o += 1
        except Exception:
            pass

    out["uniform_matches"] = ok_u / total
    out["optimal_matches"] = ok_o / total
    out["optimal_valid"] = ok_v / total
    return out
