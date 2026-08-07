import sys
import os
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from parser.core import get_optimizer_durations, get_spike_index
        got_durations = get_optimizer_durations(ref.TRACE)
        want_durations = ref.get_optimizer_durations(ref.TRACE)

        got_idx = get_spike_index(got_durations)
        want_idx = ref.get_spike_index(want_durations)

        return {
            "durations_match": 1.0 if got_durations == want_durations else 0.0,
            "index_match": 1.0 if got_idx == want_idx else 0.0
        }
    except Exception as e:
        return {"durations_match": 0.0, "index_match": 0.0, "_note": str(e)}
    finally:
        sys.path.pop(0)
