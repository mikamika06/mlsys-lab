def select_candidates(index, current_tokens, k=4):
    candidates_list = index.lookup(current_tokens)
    if not candidates_list:
        return []
    best = max(candidates_list, key=len)
    return best[:k]

def should_disable(acceptance_history, threshold=0.1):
    if not acceptance_history:
        return False
    recent = acceptance_history[-20:]
    rate = sum(recent) / len(recent)
    return rate < threshold
