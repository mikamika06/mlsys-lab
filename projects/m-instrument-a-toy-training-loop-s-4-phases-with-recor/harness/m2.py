import ref


def check(workdir):
    from profiler.trace import detect_unbalanced_ranges

    cases = ref.generate_trace_cases()
    correct = 0

    for idx, (events, want) in enumerate(cases):
        got = detect_unbalanced_ranges(events)
        if (
            isinstance(got, dict)
            and got.get("balanced") == want["balanced"]
            and got.get("error_index") == want["error_index"]
        ):
            correct += 1
        elif "_note" not in locals():
            _note = f"Case {idx} failed: got {got}, want {want}"

    acc = correct / len(cases)
    out = {"mismatch_accuracy": acc}
    if acc < 1.0 and "_note" in locals():
        out["_note"] = _note
    return out
