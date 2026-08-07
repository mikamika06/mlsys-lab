def classify_package(pkg_info):
    """Classify installed package identity and supported build flags."""
    name = pkg_info.get("name", "")
    version_str = pkg_info.get("version", "0.0.0")
    parts = version_str.split("+")[0].split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    cxx11_abi = bool(pkg_info.get("cxx11_abi", False))
    cuda_compiled = bool(pkg_info.get("cuda_compiled", False))
    has_fp8 = bool(pkg_info.get("has_fp8", False))
    has_varlen = bool(pkg_info.get("has_varlen", False))

    if name != "flash_attn" or not cuda_compiled:
        identity = "UNSUPPORTED"
    elif major >= 2 and minor >= 4:
        identity = "FLASH_ATTN_V2_ADVANCED"
    elif major >= 2:
        identity = "FLASH_ATTN_V2_BASE"
    elif major == 1:
        identity = "FLASH_ATTN_V1"
    else:
        identity = "UNSUPPORTED"

    return {
        "identity": identity,
        "major": major,
        "minor": minor,
        "cxx11_abi": cxx11_abi,
        "cuda_compiled": cuda_compiled,
        "has_fp8": has_fp8 and identity == "FLASH_ATTN_V2_ADVANCED",
        "has_varlen": has_varlen and identity in ("FLASH_ATTN_V2_BASE", "FLASH_ATTN_V2_ADVANCED"),
    }
