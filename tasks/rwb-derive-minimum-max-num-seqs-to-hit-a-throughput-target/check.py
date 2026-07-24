def _makespan(reqs, slots):
    available = [0.0] * slots
    for r in sorted(reqs, key=lambda x: x["arrival"]):
        idx = min(range(slots), key=lambda i: available[i])
        start = max(float(r["arrival"]), available[idx])
        available[idx] = start + float(r["prefill"]) + float(r["decode"])
    return max(available) if available else 0.0


def _oracle(reqs, target):
    for s in range(1, len(reqs) + 1):
        if _makespan(reqs, s) <= target:
            return s
    return -1


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                {"arrival": 0.0, "prefill": 2.0, "decode": 3.0},
                {"arrival": 1.0, "prefill": 1.0, "decode": 1.0},
                {"arrival": 2.0, "prefill": 4.0, "decode": 1.0},
            ],
            8.0,
        ),
        (
            [
                {"arrival": 0.0, "prefill": 5.0, "decode": 5.0},
                {"arrival": 0.0, "prefill": 5.0, "decode": 5.0},
                {"arrival": 0.0, "prefill": 5.0, "decode": 5.0},
                {"arrival": 0.0, "prefill": 5.0, "decode": 5.0},
            ],
            10.0,
        ),
        (
            [
                {"arrival": 0.0, "prefill": 1.5, "decode": 0.5},
                {"arrival": 10.0, "prefill": 2.0, "decode": 2.0},
                {"arrival": 11.0, "prefill": 1.0, "decode": 3.0},
                {"arrival": 12.0, "prefill": 3.0, "decode": 1.0},
            ],
            6.0,
        ),
        (
            [
                {"arrival": 0.0, "prefill": 10.0, "decode": 0.0},
                {"arrival": 0.0, "prefill": 10.0, "decode": 0.0},
                {"arrival": 0.0, "prefill": 10.0, "decode": 0.0},
            ],
            5.0,
        ),
    ]
    ok = 1.0
    for reqs, target in cases:
        expected = _oracle(reqs, target)
        try:
            got = sol.minimum_max_num_seqs([dict(x) for x in reqs], target)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
