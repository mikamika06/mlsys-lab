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
