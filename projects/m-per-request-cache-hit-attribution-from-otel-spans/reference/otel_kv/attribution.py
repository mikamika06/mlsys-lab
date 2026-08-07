def compute_attribution(parsed_requests, bytes_per_token=1024):
    res = {}
    for req in parsed_requests:
        rid = req["request_id"]
        prompt = req["prompt_tokens"]
        cached = req["cached_tokens"]
        gen = req["generated_tokens"]

        bounded_cached = min(cached, prompt) if prompt > 0 else 0
        hit_rate = (bounded_cached / float(prompt)) if prompt > 0 else 0.0
        bytes_saved = bounded_cached * bytes_per_token
        total_tokens = prompt + gen
        effective_fetched = max(0, prompt - bounded_cached)

        res[rid] = {
            "hit_rate": hit_rate,
            "bytes_saved": bytes_saved,
            "total_tokens": total_tokens,
            "effective_tokens_fetched": effective_fetched
        }
    return res
