def compute_affinity(prompt, replica_cache_state):
    if not prompt or not replica_cache_state:
        return 0.0
    matched = 0
    for token in prompt:
        if token in replica_cache_state:
            matched += 1
        else:
            break
    return matched / len(prompt)
