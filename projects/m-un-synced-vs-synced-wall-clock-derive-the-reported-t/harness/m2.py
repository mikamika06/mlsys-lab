import ref


def check(workdir):
    from timing.trace import find_missing_cuda_synchronize
    cases = ref.get_test_cases_m2()
    correct = 0
    mismatch_note = None
    for events, want in cases:
        got = find_missing_cuda_synchronize(events)
        if got == want:
            correct += 1
        elif mismatch_note is None:
            mismatch_note = f"got index {got}, want {want}"
    acc = float(correct) / float(len(cases)) if cases else 1.0
    out = {"accuracy": acc}
    if mismatch_note:
        out["_note"] = mismatch_note
    return out
