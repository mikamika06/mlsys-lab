def pack_step(token_budget: int, num_running: int, prefill_remaining: int) -> tuple:
    """Reserve one decode slot per running sequence before spending the
    remaining token budget on the pending prefill chunk."""
    decode_tokens = min(num_running, token_budget)
    prefill_chunk = min(prefill_remaining, token_budget - decode_tokens)
    return decode_tokens, prefill_chunk
