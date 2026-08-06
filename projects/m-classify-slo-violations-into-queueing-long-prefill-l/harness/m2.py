import ref


def check(workdir):
    from slo.classifier import classify_violations
    reqs, slo = ref.get_data()
    got = classify_violations(reqs, slo)
    want = ref.classify_violations(reqs, slo)
    out = {"causes_matched": 0.0}
    if got == want:
        out["causes_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
