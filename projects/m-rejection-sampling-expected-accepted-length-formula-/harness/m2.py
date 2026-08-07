import ref

def check(workdir):
    from specdec.trace import compute_acceptance_drop
    tc = ref.generate_test_case(seed=123)
    want = ref.compute_acceptance_drop_ref(tc["trace_data"])
    got = compute_acceptance_drop(tc["trace_data"])

    out = {"drop_match": 0.0, "rel_err": 1.0}
    if want == 0.0:
        rel = abs(got - want)
    else:
        rel = abs(got - want) / abs(want)

    out["rel_err"] = float(rel)
    if rel <= 1e-4:
        out["drop_match"] = 1.0
    else:
        out["_note"] = f"expected drop {want}, got {got}, rel_err {rel}"
    return out
