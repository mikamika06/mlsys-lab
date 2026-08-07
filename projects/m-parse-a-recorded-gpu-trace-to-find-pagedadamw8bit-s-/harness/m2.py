import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from parser.core import calculate_spillover_bytes
        events = ref.TRACE
        opt_events = [e for e in events if e["name"] == "optimizer_step"]
        spike_step = opt_events[7]

        got_bytes = calculate_spillover_bytes(events, spike_step["ts"], spike_step["ts"] + spike_step["dur"])
        want_bytes = ref.calculate_spillover_bytes(events, spike_step["ts"], spike_step["ts"] + spike_step["dur"])

        got_empty = calculate_spillover_bytes(events, 0, 10)
        want_empty = ref.calculate_spillover_bytes(events, 0, 10)

        match = 1.0 if (got_bytes == want_bytes and got_empty == want_empty) else 0.0
        return {"bytes_match": match}
    except Exception as e:
        return {"bytes_match": 0.0, "_note": str(e)}
    finally:
        sys.path.pop(0)
