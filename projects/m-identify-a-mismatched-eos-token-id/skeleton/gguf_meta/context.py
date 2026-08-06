def update_context_length_inplace(file_bytes: bytearray, new_ctx_len: int) -> bytearray:
    """Rewrite context_length in-place within GGUF header without altering tensor offset."""
    raise NotImplementedError
