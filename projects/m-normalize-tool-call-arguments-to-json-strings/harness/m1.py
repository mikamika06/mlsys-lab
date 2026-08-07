import ref


def check(workdir):
    from toolutils.normalize import normalize_argument

    out = {"args_matched": 0.0}
    for sample in ref.ARG_SAMPLES:
        want = ref.normalize_argument(sample)
        got = normalize_argument(sample)
        if got != want:
            out["_note"] = f"for input {sample!r}: expected {want!r}, got {got!r}"
            return out
    out["args_matched"] = 1.0
    return out
