"""Platform failure explanation."""

from fa_backend.classifier import is_backend_supported


def explain_platform_failure(backend, machine_config, input_spec):
    if is_backend_supported(backend, machine_config, input_spec):
        return "NO_FAILURE"

    cc_major, cc_minor = machine_config.get("compute_capability", (0, 0))
    head_dim = input_spec.get("head_dim", 0)
    dtype = input_spec.get("dtype", "float16")
    smem_bytes = machine_config.get("smem_per_sm_bytes", 0)

    reasons = []

    if head_dim <= 0 or head_dim % 8 != 0:
        reasons.append("INVALID_HEAD_DIM_ALIGNMENT")
    elif head_dim > 256:
        reasons.append("HEAD_DIM_TOO_LARGE")

    if backend == "FA3_HOPPER":
        if (cc_major, cc_minor) < (9, 0):
            reasons.append("UNSUPPORTED_COMPUTE_CAPABILITY")
        if dtype not in ("float16", "bfloat16", "float8"):
            reasons.append("UNSUPPORTED_DTYPE")
        if smem_bytes < 98304:
            reasons.append("INSUFFICIENT_SHARED_MEMORY")

    elif backend == "FA2_CUDA":
        if (cc_major, cc_minor) < (8, 0):
            reasons.append("UNSUPPORTED_COMPUTE_CAPABILITY")
        if dtype not in ("float16", "bfloat16"):
            reasons.append("UNSUPPORTED_DTYPE")
        if smem_bytes < 49152:
            reasons.append("INSUFFICIENT_SHARED_MEMORY")

    elif backend == "FA2_TRITON":
        if (cc_major, cc_minor) < (7, 5):
            reasons.append("UNSUPPORTED_COMPUTE_CAPABILITY")
        if dtype not in ("float16", "bfloat16"):
            reasons.append("UNSUPPORTED_DTYPE")
        if head_dim > 128 and "HEAD_DIM_TOO_LARGE" not in reasons:
            reasons.append("HEAD_DIM_TOO_LARGE")

    return "|".join(reasons) if reasons else "UNKNOWN_FAILURE"
