import ref

def check(workdir):
    from compiler_diag.guards import extract_shape_guards

    out = {"guards_matched": 0.0}
    want = ref.extract_shape_guards(ref.SAMPLE_LOGS)
    got = extract_shape_guards(ref.SAMPLE_LOGS)

    if got == want:
        out["guards_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"

    return out
