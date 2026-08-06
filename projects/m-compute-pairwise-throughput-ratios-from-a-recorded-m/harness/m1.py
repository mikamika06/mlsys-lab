import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from benchcomp.ratios import compute_pairwise_ratios, rank_frameworks

    records = ref.BENCHMARK_RECORDS
    want_ratios = ref.compute_pairwise_ratios(records)
    want_ranks = ref.rank_frameworks(records)

    got_ratios = compute_pairwise_ratios(records)
    got_ranks = rank_frameworks(records)

    out = {"ratios_rel_err": 1.0, "ranking_matches": 0.0}

    errs = []
    for pair, want_v in want_ratios.items():
        if pair not in got_ratios:
            errs.append(1.0)
        else:
            got_v = got_ratios[pair]
            errs.append(abs(got_v - want_v) / (abs(want_v) + 1e-12))

    max_err = max(errs) if errs else 1.0
    out["ratios_rel_err"] = float(max_err)

    if got_ranks == want_ranks:
        out["ranking_matches"] = 1.0
    else:
        out["_note"] = f"Expected ranking {want_ranks}, got {got_ranks}"

    return out
