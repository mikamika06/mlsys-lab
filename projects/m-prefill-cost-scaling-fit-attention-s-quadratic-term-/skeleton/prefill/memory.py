def cheapest_config(model, gpus, ctx, budget):
    """
    Finds the cheapest GPU configuration that fits the model and KV cache for ctx tokens.

    model: {"weights_gb": float, "num_layers": int, "num_kv_heads": int, "head_dim": int}
    gpus: [{"name": str, "mem_gb": int, "cost_per_hr": float, "count": int}, ...]
    ctx: int
    budget: float

    Returns (gpu_name, n_gpus) of the cheapest valid config. On cost ties, returns the one with more memory.
    """
    raise NotImplementedError
