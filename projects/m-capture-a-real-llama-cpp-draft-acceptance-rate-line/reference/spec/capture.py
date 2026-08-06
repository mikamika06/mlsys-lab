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
