import ref


def check(workdir):
    from slo.classifier import identify_violations
    reqs, slo = ref.get_data()
    got = identify_violations(reqs, slo)
    want = ref.identify_violations(reqs, slo)
    out = {"violations_matched": 0.0}
    if sorted(got) == sorted(want):
        out["violations_matched"] = float(len(want))
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
