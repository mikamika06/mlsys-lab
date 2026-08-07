from cache_analyzer.prompt_layout import format_optimized_prompt

def measure_trace_tokens(trace, use_cache_reuse=False):
    total_tokens = 0
    cached_tokens = 0
    prev_tokens = []

    for turn in trace["turns"]:
        prompt = format_optimized_prompt(
            trace["system_prompt"],
            [(t["user_message"], t["assistant_message"]) for t in trace["turns"][:turn["turn_index"]]],
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
            processed = num_tokens - common
            cached = common
        else:
            processed = num_tokens
            cached = 0

        total_tokens += processed
        cached_tokens += cached
        prev_tokens = tokens

    return {
        "processed_tokens": total_tokens,
        "cached_tokens": cached_tokens
    }
