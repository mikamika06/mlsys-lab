import numpy as np

TRACES = [
    {"draft_tokens": [10, 20, 30, 40], "accepted_tokens": [10, 20, 30]},
    {"draft_tokens": [5, 15, 25, 35, 45], "accepted_tokens": [5, 15]},
    {"draft_tokens": [100, 200, 300], "accepted_tokens": [100, 200, 300]},
    {"draft_tokens": [1, 2, 3, 4, 5, 6], "accepted_tokens": [1]}
]

PAIRS = [
    {
        "same": {"draft_tokens": [10, 20, 30, 40], "accepted_tokens": [10, 20, 30]},
        "cross": {"draft_tokens": [10, 20, 30, 40], "accepted_tokens": [10]}
    },
    {
        "same": {"draft_tokens": [1, 2, 3, 4, 5], "accepted_tokens": [1, 2, 3, 4]},
        "cross": {"draft_tokens": [1, 2, 3, 4, 5], "accepted_tokens": [1, 2]}
    }
]


def compute_acceptance_rate(trace):
    draft = trace.get("draft_tokens", [])
    accepted = trace.get("accepted_tokens", [])
    if not draft:
        return 0.0
    return float(len(accepted)) / float(len(draft))


def compare_families(same_trace, cross_trace):
    r_same = compute_acceptance_rate(same_trace)
    r_cross = compute_acceptance_rate(cross_trace)
    return float(r_same - r_cross)


def classify_pairing(trace, threshold):
    rate = compute_acceptance_rate(trace)
    return "high" if rate >= threshold else "low"


def select_best_draft(candidates):
    rates = [compute_acceptance_rate(c) for c in candidates]
    if not rates:
        return -1
    return int(np.argmax(rates))
