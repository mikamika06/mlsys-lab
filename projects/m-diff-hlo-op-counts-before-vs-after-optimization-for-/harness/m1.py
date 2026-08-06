import ref


def check(workdir):
    from hlodiff.diff import diff_op_counts

    out = {"diff_matched": 0.0}
    ok = 0
    for i, (before, after) in enumerate(ref.TEST_CASES):
        want = ref.diff_op_counts(before, after)
        got = diff_op_counts(before, after)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["diff_matched"] = float(ok)
    return out
