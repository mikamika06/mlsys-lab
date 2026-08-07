import ref


def check(workdir):
    from shimdiff.diff import diff_responses

    out = {"diffs_matched": 0.0}
    ok = 0
    total = len(ref.NATIVE_RESPONSES)

    for i in range(total):
        want = ref.diff_responses(ref.NATIVE_RESPONSES[i], ref.SHIM_RESPONSES[i])
        got = diff_responses(ref.NATIVE_RESPONSES[i], ref.SHIM_RESPONSES[i])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, expected {want}"

    if ok == total:
        out["diffs_matched"] = 1.0

    return out
