def select_chunk_size(total_tokens, budget, alpha, beta, gamma, max_chunk):
    def stall(c):
        chunks = (total_tokens + c - 1) // c
        prefill = chunks * (alpha * c * c + beta * c)
        decode = chunks * gamma * c
        return decode + max(0.0, prefill - budget)

    best = 1
    best_value = float("inf")
    for c in range(1, min(total_tokens, max_chunk) + 1):
        value = stall(c)
        if value < best_value:
            best_value = value
            best = c
    return best
