def benchmark_decode(model_fn, masker, vocab: list[str], prompt_tokens: list[int], max_tokens: int = 20) -> dict:
    """Measures decoding performance with and without schema constraint."""
    raise NotImplementedError
