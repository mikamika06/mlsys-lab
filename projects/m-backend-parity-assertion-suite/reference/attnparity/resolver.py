VALID_BACKENDS = ("auto", "eager", "sdpa", "flash_attention_2")
FLASH_DTYPES = ("float16", "bfloat16")
FLASH_HEAD_DIMS = (32, 64, 128, 256)


def resolve_attn_implementation(
    requested="auto",
    supports_flash_attn=True,
    supports_sdpa=True,
    is_right_padded=False,
    dtype="float16",
    head_dim=64,
):
    """Resolves legal attention backend implementation."""
    if requested not in VALID_BACKENDS:
        raise ValueError(f"Unknown requested backend: {requested}")

    if requested == "auto":
        if (
            supports_flash_attn
            and not is_right_padded
            and dtype in FLASH_DTYPES
            and head_dim in FLASH_HEAD_DIMS
        ):
            return "flash_attention_2"
        if supports_sdpa:
            return "sdpa"
        return "eager"

    if requested == "flash_attention_2":
        if not supports_flash_attn:
            raise ValueError("FlashAttention-2 requires hardware support.")
        if dtype not in FLASH_DTYPES:
            raise ValueError(f"FlashAttention-2 does not support dtype {dtype}.")
        if head_dim not in FLASH_HEAD_DIMS:
            raise ValueError(f"FlashAttention-2 requires head_dim in {FLASH_HEAD_DIMS}.")
        if is_right_padded:
            raise ValueError("FlashAttention-2 does not support right-padded sequences.")
        return "flash_attention_2"

    if requested == "sdpa":
        if not supports_sdpa:
            raise ValueError("SDPA requires runtime support.")
        return "sdpa"

    return "eager"
