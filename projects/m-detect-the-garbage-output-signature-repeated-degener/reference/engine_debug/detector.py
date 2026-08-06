def detect_garbage_signature(token_ids: list[int], threshold: float = 0.8) -> bool:
    if not token_ids:
        return False
    counts = {}
    for t in token_ids:
        counts[t] = counts.get(t, 0) + 1
    max_freq = max(counts.values())
    if (max_freq / len(token_ids)) >= threshold:
        return True

    n = len(token_ids)
    for length in range(1, n // 2 + 1):
        if n % length == 0:
            pattern = token_ids[:length]
            match = True
            for i in range(0, n, length):
                if token_ids[i:i+length] != pattern:
                    match = False
                    break
            if match:
                return True
    return False
