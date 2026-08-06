import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from ddpplan.trace import compute_overlap_fraction

    events = ref.generate_trace_events(seed=123)
    want = ref.compute_overlap_fraction(events)
    got = compute_overlap_fraction(events)

    diff = abs(want - got)
    matched = diff < 1e-6
    out = {"overlap_matches": 1.0 if matched else 0.0}
    return out
