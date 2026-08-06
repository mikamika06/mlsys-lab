def prefill_cost(tokens: int, cached_prefix: int, c_attn: float, c_mlp: float) -> float:
    """Return the prefill cost of a sequence given some tokens are already in the KV cache."""
    attn_pairs = (tokens * cached_prefix) + (tokens * (tokens + 1) // 2)
    return float(c_attn * attn_pairs + c_mlp * tokens)


def simulate_batch(doc_tokens: int, question_lengths: list[int], c_attn: float, c_mlp: float) -> tuple[float, float]:
    """Return (baseline_cost, cached_cost) for a batch of questions against a long document."""
    baseline = 0.0
    for q in question_lengths:
        baseline += prefill_cost(doc_tokens + q, 0, c_attn, c_mlp)

    cached = prefill_cost(doc_tokens, 0, c_attn, c_mlp)
    for q in question_lengths:
        cached += prefill_cost(q, doc_tokens, c_attn, c_mlp)

    return baseline, cached
