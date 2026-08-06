import ref


def check(workdir):
    from oomdiag.history import find_largest_allocation_site
    out = {"site_match": 0.0}
    cases = ref.REF_CASES_M2
    ok = 0
    for i, c in enumerate(cases):
        try:
            got = find_largest_allocation_site(c["snapshot"])
        except Exception as e:
            out["_note"] = f"case {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out
        if got == c["expected_site"]:
            ok += 1
    out["site_match"] = 1.0 if ok == len(cases) else 0.0
    return out
