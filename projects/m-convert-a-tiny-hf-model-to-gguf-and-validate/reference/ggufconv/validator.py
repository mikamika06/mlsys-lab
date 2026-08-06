def validate_gguf(gguf_data):
    if not isinstance(gguf_data, dict):
        return False
    if "tensors" not in gguf_data or "metadata" not in gguf_data:
        return False
    if len(gguf_data["tensors"]) == 0:
        return False
    return True
