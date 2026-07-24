import math

def _ref(kv_bpt, bw, lat, pt):
    """Closed-form break-even from the task description."""
    denom = 1.0 / pt - kv_bpt / bw
    if denom <= 0:
        return float("inf")
    if lat == 0:
        # denom > 0 means transfer is strictly cheaper per token,
        # so transfer wins for any positive L.  At L=0 both are 0.
        return 0.0
    return lat / denom

def grade(sol, fx) -> dict:
    cases = [
        # (kv_bytes_per_token, bandwidth, latency, prefill_throughput)
        (4096, 1e9, 0.001, 10_000),     # typical H100 scenario
        (8192, 2e9, 0.0005, 5_000),     # larger model, faster link
        (4096, 1e9, 0.0, 10_000),       # zero latency
        (4096, 1e9, 0.001, 100),        # very slow prefill
        (100, 1e12, 0.001, 10_000),     # absurdly fast link
        (6144, 1e9, 0.002, 8_000),      # mixed
        (2048, 5e8, 0.003, 12_000),     # slow bandwidth
        (1024, 1e10, 0.0, 15_000),      # zero latency, fast link
        (8192, 1e9, 0.001, 1_000),      # recompute always wins
        (512, 3e9, 0.005, 10_000),      # small KV, fast link
    ]

    eps = 1e-300
    max_err = 0.0

    for kv_bpt, bw, lat, pt in cases:
        ref = _ref(kv_bpt, bw, lat, pt)
        try:
            got = sol.break_even_length(kv_bpt, bw, lat, pt)
        except Exception:
            return {"rel_err": 1.0}

        if math.isinf(ref):
            # Student must also return inf (or a very large value)
            if math.isinf(got) or (isinstance(got, float) and got > 1e15):
                err = 0.0
            else:
                err = 1.0
        elif ref == 0.0:
            if got == 0.0:
                err = 0.0
            elif math.isinf(got):
                err = 1.0
            else:
                err = min(abs(got), 1.0)
        else:
            if math.isinf(got):
                err = 1.0
            else:
                err = abs(got - ref) / (abs(ref) + eps)
        max_err = max(max_err, err)

    return {"rel_err": max_err}
