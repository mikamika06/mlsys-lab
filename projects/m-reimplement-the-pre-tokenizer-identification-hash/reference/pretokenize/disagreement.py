def measure_disagreement(hf_tokens: list, gguf_tokens: list) -> float:
    if not hf_tokens and not gguf_tokens:
        return 0.0
    max_len = max(len(hf_tokens), len(gguf_tokens))
    if max_len == 0:
        return 0.0
    mismatches = sum(1 for a, b in zip(hf_tokens, gguf_tokens) if a != b)
    mismatches += abs(len(hf_tokens) - len(gguf_tokens))
    return float(mismatches) / float(max_len)
