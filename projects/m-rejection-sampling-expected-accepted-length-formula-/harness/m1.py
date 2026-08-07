import ref

def check(workdir):
    from specdec.analysis import expected_accepted_length
    tc = ref.generate_test_case(seed=123)
    want = ref.expected_accepted_length_ref(tc["probs"])
    got = expected_accepted_length(tc["probs"])

    out = {"formula_matched": 0.0, "rel_err": 1.0}
    if want == 0.0:
        rel = abs(got - want)
    else:
        rel = abs(got - want) / abs(want)

    out["rel_err"] = float(rel)
    if rel <= 1e-4:
        out["formula_matched"] = 1.0
    else:
        out["_note"] = f"expected {want}, got {got}, rel_err {rel}"
    return out
