class FallbackDiagnostics:
    def diagnose_fallback(self, kernel_config):
        dtype = kernel_config.get("dtype")
        head_dim = kernel_config.get("head_dim")
        is_contiguous = kernel_config.get("is_contiguous", True)
        alignment = kernel_config.get("alignment", 16)

        if not is_contiguous:
            return "NON_CONTIGUOUS_LAYOUT"
        if dtype not in ("float16", "bfloat16"):
            return "UNSUPPORTED_DTYPE"
        if head_dim not in (32, 64, 128, 256):
            return "INVALID_HEAD_DIM"
        if alignment % 16 != 0:
            return "MISALIGNED_ADDRESS"
        return "FLASH_ATTENTION_ELIGIBLE"
