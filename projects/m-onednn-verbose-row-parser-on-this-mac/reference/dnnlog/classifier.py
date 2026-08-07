def classify_impl(impl_str: str) -> str:
    s = impl_str.lower()
    if "jit" in s or "gemm" in s:
        if "avx" in s or "amx" in s:
            return "x86_jit"
        if "ref" in s:
            return "reference"
        return "jit_kernel"
    if "ref" in s:
        return "reference"
    return "generic"
