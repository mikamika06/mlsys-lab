import numpy as np

def compute_acceptance_drop(trace_data):
    short_ctx = np.asarray(trace_data["short_context"], dtype=np.float64)
    long_ctx = np.asarray(trace_data["long_context"], dtype=np.float64)
    rate_short = float(np.mean(short_ctx))
    rate_long = float(np.mean(long_ctx))
    if rate_short == 0.0:
        return 0.0
    drop = (rate_short - rate_long) / rate_short
    return float(drop)
