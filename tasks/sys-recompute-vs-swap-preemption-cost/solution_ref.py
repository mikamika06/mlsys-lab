import numpy as np


def modeled_mem_access(trace, bytes_per_token):
    pending = {}
    recompute = 0
    swap = 0

    for event in trace:
        if event[0] == "pause":
            pending[event[1]] = event[2]
        elif event[0] == "resume":
            req = event[1]
            if req in pending:
                tokens = pending.pop(req)
                cache_bytes = np.empty(
                    (tokens, bytes_per_token), dtype=np.uint8
                ).nbytes
                recompute += cache_bytes
                swap += cache_bytes * 2

    return {
        "recompute_bytes": int(recompute),
        "swap_bytes": int(swap),
    }
