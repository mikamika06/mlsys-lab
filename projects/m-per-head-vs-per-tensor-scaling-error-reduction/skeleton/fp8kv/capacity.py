def calc_cache_bytes(seq_len, num_heads, head_dim, num_layers, policy):
    """
    Calculate the total memory in bytes for the KV cache.
    policy can be:
      - "fp16": 2 bytes per element, no scales.
      - "fp8_per_tensor": 1 byte per element + 4 bytes for one float32 scale per layer.
      - "fp8_per_head": 1 byte per element + 4 bytes per head for float32 scales per layer.
    """
    raise NotImplementedError
