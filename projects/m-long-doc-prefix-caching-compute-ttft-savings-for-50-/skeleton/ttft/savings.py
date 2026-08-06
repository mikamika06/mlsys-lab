def prefill_cost(tokens: int, cached_prefix: int, c_attn: float, c_mlp: float) -> float:
    """Return the prefill cost of a sequence given some tokens are already in the KV cache."""
    raise NotImplementedError


def simulate_batch(doc_tokens: int, question_lengths: list[int], c_attn: float, c_mlp: float) -> tuple[float, float]:
    """Return (baseline_cost, cached_cost) for a batch of questions against a long document."""
    raise NotImplementedError
