def decode_rope_config(metadata: dict) -> dict:
    """Decode RoPE scaling configuration parameters from GGUF metadata dict."""
    arch = metadata.get("general.architecture", "llama")

    freq_base = metadata.get(f"{arch}.rope.freq_base", metadata.get("rope.freq_base", 10000.0))
    scale_type = metadata.get(f"{arch}.rope.scaling.type", metadata.get("rope.scaling.type", "none"))
    factor = metadata.get(f"{arch}.rope.scaling.factor", metadata.get("rope.scaling.factor", 1.0))
    orig_ctx = metadata.get(f"{arch}.rope.scaling.original_context_length", metadata.get("rope.scaling.original_context_length", 0))

    return {
        "freq_base": float(freq_base),
        "scaling_type": str(scale_type),
        "scaling_factor": float(factor),
        "original_context_length": int(orig_ctx)
    }
