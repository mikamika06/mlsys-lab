import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from trace_analyzer.parser import parse_events

    out = {"matches_o0": 0.0, "matches_o99": 0.0}

    t0 = ref.generate_trace_o0()
    want_t0 = ref.parse_events(t0)
    got_t0 = parse_events(t0)
    if got_t0 == want_t0:
        out["matches_o0"] = 1.0
    else:
        out["_note"] = f"O0 mismatch. want {want_t0}, got {got_t0}"

    t99 = ref.generate_trace_o99()
    want_t99 = ref.parse_events(t99)
    got_t99 = parse_events(t99)
    if got_t99 == want_t99:
        out["matches_o99"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"O99 mismatch. want {want_t99}, got {got_t99}"

    return out
