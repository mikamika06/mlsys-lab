import ref


def check(workdir):
    from onnxtools.analyzer import find_initializer_inputs

    model = ref.generate_test_case_1()
    got = sorted(find_initializer_inputs(model))
    want = ["w1"]
    out = {"overlaps_matched": 0.0}
    if got == want:
        out["overlaps_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
