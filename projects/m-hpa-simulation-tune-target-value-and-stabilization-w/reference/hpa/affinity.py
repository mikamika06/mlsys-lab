def quantify_hit_rate_loss(session_turns, num_replicas):
    if not session_turns:
        return 0.0
    total = sum(session_turns)
    if total == 0:
        return 0.0
    possible = sum(t - 1 for t in session_turns if t > 1)
    perfect = possible / total
    rand = (possible / num_replicas) / total
    return perfect - rand
