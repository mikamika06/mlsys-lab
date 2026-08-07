def check_hardware_gating(pkg_identity, hw_info):
    """Determine if host hardware meets kernel prerequisites."""
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
