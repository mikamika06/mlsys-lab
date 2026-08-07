def classify_workload_dominance(ttft, prefill_tok_per_sec, decode_tok_per_sec, prompt_tokens, completion_tokens):
    prefill_time = ttft
    decode_time = (completion_tokens - 1) / decode_tok_per_sec if decode_tok_per_sec > 0 else 0.0

    if prefill_time >= decode_time:
        return "prefill-dominated"
    return "decode-dominated"
