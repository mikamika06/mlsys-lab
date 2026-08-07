SUPPORTED_MATRIX = {
    "W4A16": ["ampere", "hopper", "blackwell"],
    "W8A8": ["ampere", "hopper", "blackwell"],
    "W4A8": ["hopper", "blackwell"],
    "FP8": ["hopper", "blackwell"],
    "INT4": ["ampere", "hopper", "blackwell"],
    "INT8": ["ampere", "hopper", "blackwell"],
}


def get_supported_architectures(scheme_name: str) -> list:
    return SUPPORTED_MATRIX.get(scheme_name.upper(), [])


def has_native_kernel(scheme_name: str, arch: str) -> bool:
    supported = get_supported_architectures(scheme_name)
    return arch.lower() in supported
