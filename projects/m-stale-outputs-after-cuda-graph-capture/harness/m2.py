import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    try:
        from ortgraph.arena import analyze_arena_vs_rss
        from ortgraph.capture import detect_stale_outputs, simulate_capture_and_run
    except Exception as e:
        return {
            "stale_detected": 0.0,
            "arena_profile_matched": 0.0,
            "_note": f"Import failed: {e}",
        }

    out = {"stale_detected": 0.0, "arena_profile_matched": 0.0}

    try:
        cap_res = simulate_capture_and_run(ref.EXECUTION_STEPS, ref.REPLAY_INPUTS)
        for entry in cap_res:
            sid = entry["step_id"]
            if sid in ref.EXPECTED_STEP_OUTPUTS:
                entry["expected"] = ref.EXPECTED_STEP_OUTPUTS[sid]

        got_stale = detect_stale_outputs(cap_res)
        want_stale = ref.detect_stale_outputs(cap_res)

        if sorted(got_stale) == sorted(want_stale) and len(want_stale) > 0:
            out["stale_detected"] = 1.0
        else:
            out["_note"] = f"Stale outputs mismatch: got {got_stale}, expected {want_stale}"
    except Exception as e:
        out["_note"] = f"Capture/stale simulation failed: {e}"
        return out

    try:
        got_arena = analyze_arena_vs_rss(ref.ALLOCATIONS, ref.DEALLOCATIONS, ref.BLOCK_SIZE)
        want_arena = ref.analyze_arena_vs_rss(ref.ALLOCATIONS, ref.DEALLOCATIONS, ref.BLOCK_SIZE)

        if got_arena == want_arena:
            out["arena_profile_matched"] = 1.0
        else:
            out["_note"] = f"Arena profile mismatch: got {got_arena}, expected {want_arena}"
    except Exception as e:
        out["_note"] = f"Arena analysis failed: {e}"

    return out
