def pack_step(token_budget: int, num_running: int, prefill_remaining: int) -> tuple:
    """Pack one scheduling step: split token_budget between decode and the
    pending chunked-prefill request."""
    # BUG: spends the whole budget on the prefill chunk first, so decode can
    # be starved down to zero whenever the prefill chunk alone exhausts the
    # budget.
    prefill_chunk = min(prefill_remaining, token_budget)
    decode_tokens = min(num_running, token_budget - prefill_chunk)
    return decode_tokens, prefill_chunk
