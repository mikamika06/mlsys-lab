def select_quantization(tensor_shapes: dict[str, tuple[int, ...]],
                        quants: dict[str, float],
                        context_length: int,
                        num_layers: int,
                        num_kv_heads: int,
                        head_dim: int,
                        vram_budget_bytes: int) -> dict:
    raise NotImplementedError
