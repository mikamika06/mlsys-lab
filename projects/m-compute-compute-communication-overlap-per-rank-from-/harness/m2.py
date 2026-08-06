import ref

def check(workdir):
    from prof.straggler import identify_straggler
    trace_data, _, _ = ref.generate_fixtures()
    out = {"straggler_matched": 0.0}
    try:
        got = identify_straggler(trace_data)
        want = ref.ref_identify_straggler(trace_data)
        if got == want:
            out["straggler_matched"] = 1.0
        else:
            out["_note"] = f"straggler mismatch: got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
