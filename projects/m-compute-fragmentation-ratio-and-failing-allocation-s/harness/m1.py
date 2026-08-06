import ref


def check(workdir):
    from oomdiag.analysis import compute_fragmentation_and_failing_size
    out = {"fragmentation_match": 0.0, "failing_size_match": 0.0}
    cases = ref.REF_CASES_M1
    frag_ok = 0
    size_ok = 0
    for i, c in enumerate(cases):
        try:
            got_frag, got_req = compute_fragmentation_and_failing_size(c["dump"])
        except Exception as e:
            out["_note"] = f"case {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out
        if abs(got_frag - c["expected_frag"]) < 1e-5:
            frag_ok += 1
        if got_req == c["expected_req"]:
            size_ok += 1
    out["fragmentation_match"] = 1.0 if frag_ok == len(cases) else 0.0
    out["failing_size_match"] = 1.0 if size_ok == len(cases) else 0.0
    return out
