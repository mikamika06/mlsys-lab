import ref


def check(workdir):
    from roofline.predict import max_gflops

    out = {"predictions_matched": 0.0}
    correct = 0
    total = len(ref.TEST_CASES_M2)
    for tc in ref.TEST_CASES_M2:
        got = max_gflops(tc["ai"], tc["peak_gflops"], tc["peak_bw"])
        if abs(got - tc["want"]) < 1e-5:
            correct += 1
        else:
            out["_note"] = f"ai {tc['ai']}: got {got}, want {tc['want']}"
    if correct == total:
        out["predictions_matched"] = 1.0
    return out
