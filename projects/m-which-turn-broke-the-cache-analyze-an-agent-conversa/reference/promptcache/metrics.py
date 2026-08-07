from promptcache.layout import format_optimized_prompt

def measure_trace_tokens(trace, use_cache_reuse=False):
    total_processed = 0
    total_cached = 0
    prev_tokens = []

    for turn in trace["turns"]:
        history = [(t["user_message"], t["assistant_message"]) for t in trace["turns"][:turn["turn_index"]]]
        prompt = format_optimized_prompt(
            trace["system_prompt"],
            history,
            turn["user_message"],
            turn["dynamic_state"]
        )
        tokens = prompt.split()
        num_tokens = len(tokens)

        if use_cache_reuse and prev_tokens:
            common = 0
            min_len = min(len(prev_tokens), len(tokens))
            for a, b in zip(prev_tokens[:min_len], tokens[:min_len]):
                if a == b:
                    common += 1
                else:
                    break
            cached = common
            processed = num_tokens - common
        else:
            cached = 0
            processed = num_tokens

        total_processed += processed
        total_cached += cached
        prev_tokens = tokens

    return {
        "processed_tokens": total_processed,
        "cached_tokens": total_cached
    }
