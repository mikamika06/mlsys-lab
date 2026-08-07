from fa_classifier.classifier import classify_package
from fa_classifier.gating import check_hardware_gating


def resolve_dispatch_target(pkg_info, hw_info):
    """Resolve the execution target and status for the given package and hardware."""
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
