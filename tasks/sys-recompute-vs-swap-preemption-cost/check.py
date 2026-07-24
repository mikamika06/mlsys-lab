import numpy as np


def _oracle(trace, bytes_per_token):
    pending = {}
    recompute = 0
    swap = 0

    def cache_bytes(tokens):
        return np.empty((tokens, bytes_per_token), dtype=np.uint8).nbytes

    for event in trace:
        if event[0] == "pause":
            pending[event[1]] = event[2]
        elif event[0] == "resume":
            req = event[1]
            if req in pending:
                size = cache_bytes(pending.pop(req))
                recompute += size
                swap += size * 2
    return {
        "recompute_bytes": int(recompute),
        "swap_bytes": int(swap),
    }


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                ("pause", "a", 10),
                ("resume", "a"),
            ],
            4,
        ),
        (
            [
                ("pause", "x", 3),
                ("pause", "y", 7),
                ("resume", "x"),
                ("resume", "y"),
            ],
            16,
        ),
        (
            [
                ("resume", "missing"),
                ("pause", 11, 0),
                ("resume", 11),
                ("pause", 12, 5),
                ("resume", 12),
            ],
            32,
        ),
        (
            [
                ("pause", "same", 2),
                ("pause", "same", 8),
                ("resume", "same"),
                ("resume", "same"),
            ],
            5,
        ),
    ]

    ok = 1.0
    for trace, bpt in cases:
        try:
            got = sol.modeled_mem_access(trace, bpt)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(trace, bpt):
            ok = 0.0
            break
    return {"modeled_mem_access": ok}
