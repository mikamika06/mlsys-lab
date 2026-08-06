def compute_acceptance_rate(draft_tokens, target_tokens):
    if not draft_tokens:
        return 0.0
    accepted = 0
    n = min(len(draft_tokens), len(target_tokens))
    for i in range(n):
        if draft_tokens[i] == target_tokens[i]:
            accepted += 1
        else:
            break
    return accepted / len(draft_tokens)


def evaluate_pairings(pairings_data):
    results = {}
    for name, data in pairings_data.items():
        rate = compute_acceptance_rate(data["draft_tokens"], data["target_tokens"])
        results[name] = rate
    return results
