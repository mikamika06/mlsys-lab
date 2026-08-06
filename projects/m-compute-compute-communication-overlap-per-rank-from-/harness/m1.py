import ref

def check(workdir):
    from prof.overlap import compute_overlap
    trace_data, _, _ = ref.generate_fixtures()
    out = {"overlaps_matched": 0.0}
    try:
        got = compute_overlap(trace_data)
        want = ref.ref_compute_overlap(trace_data)
        if isinstance(got, dict) and set(got.keys()) == set(want.keys()):
            all_close = True
            for k in want:
                if abs(float(got[k]) - float(want[k])) > 1e-4:
                    all_close = False
                    break
            if all_close:
                out["overlaps_matched"] = 1.0
            else:
                out["_note"] = f"mismatch: got {got}, want {want}"
        else:
            out["_note"] = f"keys mismatch or non-dict: got {got}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
