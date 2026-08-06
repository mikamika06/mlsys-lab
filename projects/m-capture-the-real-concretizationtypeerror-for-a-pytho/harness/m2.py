import numpy as np
import ref

def check(workdir):
    from jaxcomp.retrace import trace_and_count_retraces

    out = {"retrace_counts_matched": 0.0}

    def sample_fn(x):
        return x

    inputs = [
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((2, 8)),
        np.zeros((2, 8)),
    ]

    got_count = trace_and_count_retraces(sample_fn, inputs)
    ref_count = ref.trace_and_count_retraces(sample_fn, inputs)

    if got_count == ref_count and got_count == 3:
        out["retrace_counts_matched"] = 1.0
    else:
        out["_note"] = f"Expected {ref_count} retraces for 3 distinct shapes, got {got_count}"

    return out
