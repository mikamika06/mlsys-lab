import sys
import os
import ref

sys.path.insert(0, os.path.abspath("."))


def check(workdir):
    from nsys_diag.nvtx import reconstruct_nvtx_depths, analyze_nvtx_nesting
    import reference.nsys_diag.nvtx as ref_nvtx

    out = {"nvtx_traces_matched": 0, "total_traces": 10}
    ok = 0

    for i in range(10):
        events = ref.generate_nvtx_events(seed=200 + i)

        want_timeline = ref_nvtx.reconstruct_nvtx_depths(events)
        got_timeline = reconstruct_nvtx_depths(events)

        want_analysis = ref_nvtx.analyze_nvtx_nesting(events)
        got_analysis = analyze_nvtx_nesting(events)

        timeline_ok = len(want_timeline) == len(got_timeline) and all(
            w["timestamp_ns"] == g.get("timestamp_ns") and
            w["event_type"] == g.get("event_type") and
            w["depth"] == g.get("depth")
            for w, g in zip(want_timeline, got_timeline)
        )

        analysis_ok = (
            want_analysis["max_depth"] == got_analysis.get("max_depth") and
            want_analysis["duration_by_depth_ns"] == got_analysis.get("duration_by_depth_ns")
        )

        if timeline_ok and analysis_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"Trace {i} mismatch: want {want_analysis}, got {got_analysis}"

    out["nvtx_traces_matched"] = ok
    return out
