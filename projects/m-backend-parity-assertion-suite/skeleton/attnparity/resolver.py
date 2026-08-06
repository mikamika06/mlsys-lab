def resolve_attn_implementation(
    requested="auto",
    supports_flash_attn=True,
    supports_sdpa=True,
    is_right_padded=False,
    dtype="float16",
    head_dim=64,
):
    """Resolves legal attention backend implementation."""
    raise NotImplementedError
