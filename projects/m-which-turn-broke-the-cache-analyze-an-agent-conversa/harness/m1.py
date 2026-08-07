import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from cacheplan.analyze import simulate_processing, find_breaking_turn

    counts = simulate_processing(ref.BAD_LOG)
    want = ref.simulate_processing(ref.BAD_LOG)

    err = sum(abs(g - w) for g, w in zip(counts, want))
    rel_err = err / max(sum(want), 1)

    broke = find_breaking_turn(counts)
    want_broke = ref.find_breaking_turn(want)

    return {
        "simulate_rel_err": float(rel_err),
        "broke_match": 1.0 if broke == want_broke else 0.0
    }
