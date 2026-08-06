import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from prefixcache.dynamic import DynamicCache
    from prefixcache.priming import prime_prefix, reuse_prefix_priming

    out = {"prefix_matched": 0.0, "cache_reused": 0.0, "tokens_correct": 0.0}

    model = ref.SimulatedModel(num_layers=2, num_heads=4, head_dim=16, vocab_size=100)
    prefix = ref.PREFIX_TOKENS

    try:
        prefix_cache, logits = prime_prefix(prefix, model.forward)
    except Exception as e:
        out["_note"] = f"prime_prefix raised exception: {type(e).__name__}: {str(e)}"
        return out

    if not isinstance(prefix_cache, DynamicCache):
        out["_note"] = "prime_prefix did not return a DynamicCache instance"
        return out

    if prefix_cache.get_seq_length() == len(prefix):
        out["prefix_matched"] = 1.0
    else:
        out["_note"] = (
            f"prefix length {prefix_cache.get_seq_length()} != expected {len(prefix)}"
        )
        return out

    prompts = ref.TEST_PROMPTS
    prefix_len_before = prefix_cache.get_seq_length()

    try:
        results = reuse_prefix_priming(prefix_cache, prompts, model.forward)
    except Exception as e:
        out["_note"] = (
            f"reuse_prefix_priming raised exception: {type(e).__name__}: {str(e)}"
        )
        return out

    prefix_len_after = prefix_cache.get_seq_length()

    if prefix_len_before == prefix_len_after:
        out["cache_reused"] = 1.0
    else:
        out["_note"] = (
            "prefix_cache was mutated during reuse_prefix_priming (should be cloned)"
        )
        return out

    all_ok = True
    for i, (p_tokens, (res_logits, res_cache)) in enumerate(zip(prompts, results)):
        expected_len = len(prefix) + len(p_tokens)
        if res_cache.get_seq_length() != expected_len:
            all_ok = False
            out["_note"] = (
                f"prompt {i} cache length {res_cache.get_seq_length()} != expected {expected_len}"
            )
            break

    if all_ok and len(results) == len(prompts):
        out["tokens_correct"] = 1.0

    return out
