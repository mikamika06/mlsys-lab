def prime_prefix(prefix_tokens, model_fn):
    from prefixcache.dynamic import DynamicCache

    cache = DynamicCache()
    logits, cache = model_fn(prefix_tokens, past_key_values=cache, start_pos=0)
    return cache, logits


def reuse_prefix_priming(prefix_cache, prompt_tokens_list, model_fn):
    results = []
    for prompt in prompt_tokens_list:
        cloned = prefix_cache.copy()
        start_pos = cloned.get_seq_length()
        logits, updated_cache = model_fn(
            prompt, past_key_values=cloned, start_pos=start_pos
        )
        results.append((logits, updated_cache))
    return results
