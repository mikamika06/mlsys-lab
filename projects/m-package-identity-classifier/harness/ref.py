def classify_package(pkg_info):
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


def check_hardware_gating(pkg_identity, hw_info):
    sm_major = hw_info.get("sm_major", 0)
    sm_minor = hw_info.get("sm_minor", 0)
    compute_capability = sm_major * 10 + sm_minor
    has_bf16 = hw_info.get("has_bf16", False)

    if pkg_identity == "FLASH_ATTN_V2_ADVANCED":
        compatible = compute_capability >= 80 and has_bf16
        min_sm = 80
    elif pkg_identity == "FLASH_ATTN_V2_BASE":
        compatible = compute_capability >= 80
        min_sm = 80
    elif pkg_identity == "FLASH_ATTN_V1":
        compatible = compute_capability >= 75
        min_sm = 75
    else:
        compatible = False
        min_sm = 0

    return {
        "compatible": compatible,
        "compute_capability": compute_capability,
        "min_sm_required": min_sm,
    }


def resolve_dispatch_target(pkg_info, hw_info):
    pkg_meta = classify_package(pkg_info)
    gate_meta = check_hardware_gating(pkg_meta["identity"], hw_info)

    if not pkg_meta["cuda_compiled"] or pkg_meta["identity"] == "UNSUPPORTED":
        status = "REJECTED_BUILD"
        kernel = "CPU_REFERENCE"
    elif not gate_meta["compatible"]:
        status = "REJECTED_HARDWARE"
        kernel = "TRITON_FALLBACK"
    else:
        status = "ACCEPTED"
        if pkg_meta["identity"] == "FLASH_ATTN_V2_ADVANCED":
            kernel = "FA2_CUDNN_FLASH" if pkg_meta["has_fp8"] and hw_info.get("sm_major", 0) >= 9 else "FA2_CUTLASS_FLASH"
        elif pkg_meta["identity"] == "FLASH_ATTN_V2_BASE":
            kernel = "FA2_CUTLASS_FLASH"
        else:
            kernel = "FA1_CUDA_FLASH"

    return {
        "status": status,
        "target_kernel": kernel,
        "pkg_identity": pkg_meta["identity"],
        "compatible": gate_meta["compatible"],
    }


TEST_PACKAGES = [
    {"name": "flash_attn", "version": "2.5.2+cu122", "cxx11_abi": True, "cuda_compiled": True, "has_fp8": True, "has_varlen": True},
    {"name": "flash_attn", "version": "2.1.0", "cxx11_abi": False, "cuda_compiled": True, "has_fp8": True, "has_varlen": True},
    {"name": "flash_attn", "version": "1.0.9", "cxx11_abi": True, "cuda_compiled": True, "has_fp8": False, "has_varlen": False},
    {"name": "flash_attn_custom", "version": "2.5.0", "cxx11_abi": True, "cuda_compiled": True, "has_fp8": True, "has_varlen": True},
    {"name": "flash_attn", "version": "2.5.0", "cxx11_abi": True, "cuda_compiled": False, "has_fp8": True, "has_varlen": True},
]

TEST_HARDWARE = [
    {"sm_major": 9, "sm_minor": 0, "has_bf16": True},
    {"sm_major": 8, "sm_minor": 0, "has_bf16": True},
    {"sm_major": 8, "sm_minor": 0, "has_bf16": False},
    {"sm_major": 7, "sm_minor": 5, "has_bf16": False},
    {"sm_major": 7, "sm_minor": 0, "has_bf16": False},
]
