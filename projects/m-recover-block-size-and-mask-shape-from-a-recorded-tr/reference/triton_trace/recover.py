import numpy as np


def recover_block_size(trace_events):
    offsets = [e["offset"] for e in trace_events if "offset" in e]
    if not offsets:
        return (1, 1)
    diffs = np.diff(np.unique(offsets))
    if len(diffs) == 0:
        return (1, len(offsets))
    step = int(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 1
    total = len(offsets)
    block_x = int(np.gcd(total, step * 16))
    block_y = max(1, total // max(1, block_x))
    return (block_x, block_y)
