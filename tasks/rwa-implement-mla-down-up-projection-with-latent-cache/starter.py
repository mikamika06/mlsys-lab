import math

def mla_forward(x: list[list[float]], W_Q: list[list[float]], W_down_kv: list[list[float]], W_up_K: list[list[float]], W_up_V: list[list[float]], num_heads: int) -> tuple[list[list[float]], list[list[float]]]:
    """Multi-head Latent Attention forward pass with a shared low-rank KV cache.

    x: (n, d_model) float64 hidden states.
    W_Q: (d_model, num_heads * d_head) query projection.
    W_down_kv: (d_model, kv_lora_rank) shared KV down-projection.
    W_up_K, W_up_V: (kv_lora_rank, num_heads * d_head) per-head up-projections.
    num_heads: number of attention heads.

    Returns (out, c_kv):
      c_kv: (n, kv_lora_rank) cached latent, x @ W_down_kv -- the only
        thing a real KV cache would need to keep.
      out: (n, num_heads * d_head) multi-head self-attention output,
        computed by up-projecting K, V from c_kv and running standard
        (non-causal) scaled dot-product attention per head.
    """
    raise NotImplementedError('your code here')
