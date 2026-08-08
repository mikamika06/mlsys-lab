def compute_acceptance_rate(trace):
    draft = trace.get("draft_tokens", [])
    accepted = trace.get("accepted_tokens", [])
    if not draft:
        return 0.0
    return float(len(accepted)) / float(len(draft))
