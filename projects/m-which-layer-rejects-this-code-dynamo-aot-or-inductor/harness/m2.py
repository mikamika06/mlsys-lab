import ref


def check(workdir):
    from compilerdiag import repro, decompositions

    out = {"repro_extracted": 0.0, "decomposition_count_matched": 0.0}
    repro_ok = 0
    decomp_ok = 0

    for s in ref.SCENARIOS:
        got_repro = repro.extract_repro(s["id"])
        if isinstance(got_repro, str) and len(got_repro.strip()) > 0:
            repro_ok += 1

        got_count = decompositions.count_decompositions(s["id"])
        if got_count == s["decomposition_count"]:
            decomp_ok += 1

    out["repro_extracted"] = 1.0 if repro_ok == len(ref.SCENARIOS) else 0.0
    out["decomposition_count_matched"] = 1.0 if decomp_ok == len(ref.SCENARIOS) else 0.0
    return out
