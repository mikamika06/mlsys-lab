def simulate_rss(tokens: int, cache_limit: int | None, model_bytes: int, token_bytes: int) -> list[int]:
    out = []
    current_cache = 0
    for _ in range(tokens):
        current_cache += token_bytes
        if cache_limit is not None and current_cache > cache_limit:
            current_cache = cache_limit
        out.append(model_bytes + current_cache)
    return out
