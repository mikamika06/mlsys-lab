CONFIGS = [
    [
        {"draft_len": 4, "n_accepted": 3},
        {"draft_len": 4, "n_accepted": 2},
        {"draft_len": 4, "n_accepted": 4},
    ],
    [
        {"draft_len": 8, "n_accepted": 5},
        {"draft_len": 8, "n_accepted": 6},
    ],
    [
        {"draft_len": 2, "n_accepted": 1},
        {"draft_len": 2, "n_accepted": 2},
        {"draft_len": 2, "n_accepted": 0},
    ]
]

def capture_acceptance_rate(timings, n_max):
    accepted = 0
    proposed = 0
    for t in timings:
        draft_len = min(t.get("draft_len", 0), n_max)
        n_acc = t.get("n_accepted", 0)
        proposed += draft_len
        accepted += min(n_acc, draft_len)
    ratio = accepted / proposed if proposed > 0 else 0.0
    return {"proposed": proposed, "accepted": accepted, "acceptance_rate": ratio}

def find_optimal_draft_n(eval_func, n_values):
    best_n = None
    best_throughput = -1.0
    results = {}
    for n in n_values:
        throughput = eval_func(n)
        results[n] = throughput
        if throughput > best_throughput:
            best_throughput = throughput
            best_n = n
    return {"optimal_n": best_n, "optimal_throughput": best_throughput, "results": results}

def recompute_draft_accept_ratio(timings_per_token):
    total_draft = 0
    total_acc = 0
    for entry in timings_per_token:
        d = entry.get("draft_length", 0)
        a = entry.get("accepted_count", 0)
        total_draft += d
        total_acc += min(a, d)
    if total_draft == 0:
        return 0.0
    return total_acc / total_draft
