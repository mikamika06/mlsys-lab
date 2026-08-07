def get_supported_schemes(arch):
    arch = arch.lower()
    if "ampere" in arch:
        return ["w8a16", "fp8"]
    elif "hopper" in arch:
        return ["w8a16", "w4a16", "fp8"]
    elif "blackwell" in arch:
        return ["w8a16", "w4a16", "fp8", "int4"]
    return ["w8a16"]
