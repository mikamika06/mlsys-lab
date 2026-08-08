import ref


def check(workdir):
    from nvtxprof.mac import analyze_mac_trace

    out = {"rankings_correct": 0.0, "metrics_matched": 0.0}
    want = ref.analyze_mac_trace(ref.MAC_TRACE_EVENTS, ref.TARGET_PHASES)
    try:
        got = analyze_mac_trace(ref.MAC_TRACE_EVENTS, ref.TARGET_PHASES)
    except Exception as e:
        out["_note"] = f"analyze_mac_trace raised {type(e).__name__}: {e}"
        return out

    if got is None or not isinstance(got, dict):
        out["_note"] = "analyze_mac_trace returned invalid structure"
        return out

    if got.get("rankings") == want.get("rankings"):
        out["rankings_correct"] = 1.0

    if got.get("phase_metrics") == want.get("phase_metrics"):
        out["metrics_matched"] = 1.0
    else:
        out["_note"] = f"phase_metrics mismatch: got {got.get('phase_metrics')}, want {want.get('phase_metrics')}"

    return out
