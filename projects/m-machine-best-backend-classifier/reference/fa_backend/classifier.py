"""Machine to backend classifier."""

VALID_BACKENDS = ["FA3_HOPPER", "FA2_TRITON", "FA2_CUDA", "MATH_FALLBACK"]


def is_backend_supported(backend, machine_config, input_spec):
    cc_major, cc_minor = machine_config.get("compute_capability", (0, 0))
    head_dim = input_spec.get("head_dim", 0)
    dtype = input_spec.get("dtype", "float16")
    smem_bytes = machine_config.get("smem_per_sm_bytes", 0)

    if head_dim <= 0 or head_dim % 8 != 0 or head_dim > 256:
        return False

    if backend == "FA3_HOPPER":
        if (cc_major, cc_minor) < (9, 0):
            return False
        if dtype not in ("float16", "bfloat16", "float8"):
            return False
        if smem_bytes < 98304:
            return False
        return True

    if backend == "FA2_CUDA":
        if (cc_major, cc_minor) < (8, 0):
            return False
        if dtype not in ("float16", "bfloat16"):
            return False
        if smem_bytes < 49152:
            return False
        return True

    if backend == "FA2_TRITON":
        if (cc_major, cc_minor) < (7, 5):
            return False
        if dtype not in ("float16", "bfloat16"):
            return False
        if head_dim > 128:
            return False
        return True

    if backend == "MATH_FALLBACK":
        return True

    return False


def classify_backend(machine_config, input_spec):
    candidates = ["FA3_HOPPER", "FA2_CUDA", "FA2_TRITON", "MATH_FALLBACK"]
    for b in candidates:
        if is_backend_supported(b, machine_config, input_spec):
            return b
    return "MATH_FALLBACK"
