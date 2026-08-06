import ref


def check(workdir):
    from snaptool.timeline import build_timeline

    out = {"timeline_matched": 0.0, "peak_bytes_matched": 0.0}
    snapshot = ref.generate_mock_snapshot(seed=123)
    
    want_timeline, want_peak = ref.compute_reference_timeline(snapshot)
    
    try:
        got_timeline, got_peak = build_timeline(snapshot)
    except Exception as e:
        out["_note"] = f"build_timeline raised exception: {type(e).__name__}: {e}"
        return out

    if got_timeline == want_timeline:
        out["timeline_matched"] = 1.0
    else:
        out["_note"] = f"timeline mismatch. expected len {len(want_timeline)}, got len {len(got_timeline)}"

    if got_peak == want_peak:
        out["peak_bytes_matched"] = 1.0
    else:
        out["_note"] = f"peak bytes mismatch. expected {want_peak}, got {got_peak}"

    return out
