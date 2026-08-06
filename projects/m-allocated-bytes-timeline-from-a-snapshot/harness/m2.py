import ref


def check(workdir):
    from snaptool.frames import find_retaining_frame
    from snaptool.footprint import compare_footprint

    out = {"retaining_frame_matched": 0.0, "leak_bytes_matched": 0.0}
    snapshot = ref.generate_mock_snapshot(seed=456)
    
    want_frame, want_bytes = ref.compute_reference_retaining_frame(snapshot)
    
    try:
        got_frame, got_bytes = find_retaining_frame(snapshot)
    except Exception as e:
        out["_note"] = f"find_retaining_frame raised exception: {type(e).__name__}: {e}"
        return out

    if got_frame == want_frame:
        out["retaining_frame_matched"] = 1.0
    if got_bytes == want_bytes:
        out["leak_bytes_matched"] = 1.0

    if out["retaining_frame_matched"] == 0.0 or out["leak_bytes_matched"] == 0.0:
        out["_note"] = f"Expected ({want_frame}, {want_bytes}), got ({got_frame}, {got_bytes})"
        return out

    try:
        want_theo, want_over = ref.compute_reference_footprint(snapshot)
        got_theo, got_over = compare_footprint(snapshot)
        if got_theo != want_theo or got_over != want_over:
            out["_note"] = f"Footprint mismatch. Expected ({want_theo}, {want_over}), got ({got_theo}, {got_over})"
            out["leak_bytes_matched"] = 0.0
    except Exception as e:
        out["_note"] = f"compare_footprint raised exception: {type(e).__name__}: {e}"
        out["leak_bytes_matched"] = 0.0

    return out
