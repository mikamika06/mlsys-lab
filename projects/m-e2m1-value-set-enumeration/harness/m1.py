import ref

def check(workdir):
    from e2m1.enumeration import enumerate_e2m1
    out = {"enumeration_matched": 0.0}
    try:
        got = enumerate_e2m1()
        want = ref.enumerate_e2m1()
        if got == want:
            out["enumeration_matched"] = 1.0
        else:
            out["_note"] = "enumeration mismatch"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}"
    return out
