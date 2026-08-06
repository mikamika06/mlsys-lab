def parse_gguf(data: bytes) -> tuple:
    """Parse GGUF file bytes into header, metadata, tensor info, and payload."""
    raise NotImplementedError


def add_or_update_chat_template(gguf_bytes: bytes, template_str: str) -> bytes:
    """Inject or update tokenizer.chat_template metadata in GGUF bytes."""
    raise NotImplementedError
