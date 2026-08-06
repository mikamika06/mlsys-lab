def compute_padding_waste(requests, max_batch_size):
    """
    Computes token padding waste for static vs continuous batching.

    requests: list of dicts with 'prompt_len' and 'decode_len'
    max_batch_size: max requests allowed in a batch simultaneously

    Returns dict with:
      - static_padded_tokens: int
      - static_useful_tokens: int
      - static_waste_ratio: float
      - continuous_padded_tokens: int
      - continuous_useful_tokens: int
      - continuous_waste_ratio: float
    """
    raise NotImplementedError
