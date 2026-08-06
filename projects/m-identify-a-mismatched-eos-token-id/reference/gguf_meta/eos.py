def find_mismatched_eos(metadata: dict) -> dict:
    """Identify mismatched EOS token IDs across metadata keys."""
    arch = metadata.get("general.architecture", "llama")
    arch_eos_key = f"{arch}.tokenizer.eos_token_id"

    arch_eos = metadata.get(arch_eos_key)
    if arch_eos is None:
        arch_eos = metadata.get("tokenizer.ggml.eos_token_id")

    vocab_eos = None
    tokens = metadata.get("tokenizer.ggml.tokens", [])
    scores = metadata.get("tokenizer.ggml.scores", [])
    token_types = metadata.get("tokenizer.ggml.token_type", [])

    for i, t_type in enumerate(token_types):
        if t_type == 3:
            vocab_eos = i
            break

    mismatch = False
    if arch_eos is not None and vocab_eos is not None and arch_eos != vocab_eos:
        mismatch = True

    return {
        "arch_eos_id": arch_eos,
        "vocab_eos_id": vocab_eos,
        "mismatch": mismatch
    }
