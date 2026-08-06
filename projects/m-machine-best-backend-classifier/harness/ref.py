"""Oracle reference definitions for verification harness."""

MACHINES = [
    {
        "id": "h100_sxm",
        "compute_capability": (9, 0),
        "smem_per_sm_bytes": 233472,
        "sm_count": 132,
        "clock_ghz": 1.75,
    },
    {
        "id": "a100_80gb",
        "compute_capability": (8, 0),
        "smem_per_sm_bytes": 167936,
        "sm_count": 108,
        "clock_ghz": 1.41,
    },
    {
        "id": "t4_gpu",
        "compute_capability": (7, 5),
        "smem_per_sm_bytes": 65536,
        "sm_count": 40,
        "clock_ghz": 1.59,
    },
    {
        "id": "v100_sxm",
        "compute_capability": (7, 0),
        "smem_per_sm_bytes": 98304,
        "sm_count": 80,
        "clock_ghz": 1.53,
    },
]

INPUT_SPECS = [
    {
        "id": "standard_llama",
        "batch_size": 2,
        "seq_len": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "dtype": "bfloat16",
    },
    {
        "id": "fp8_hopper",
        "batch_size": 4,
        "seq_len": 2048,
        "num_heads": 64,
        "head_dim": 64,
        "dtype": "float8",
    },
    {
        "id": "misaligned_head",
        "batch_size": 1,
        "seq_len": 1024,
        "num_heads": 16,
        "head_dim": 100,
        "dtype": "float16",
    },
    {
        "id": "large_head",
        "batch_size": 1,
        "seq_len": 512,
        "num_heads": 8,
        "head_dim": 256,
        "dtype": "float16",
    },
]

TARGET_BACKENDS = ["FA3_HOPPER", "FA2_CUDA", "FA2_TRITON"]


def ref_classify(machine, input_spec):
    cc_major, cc_minor = machine.get("compute_capability", (0, 0))
    head_dim = input_spec.get("head_dim", 0)
    dtype = input_spec.get("dtype", "float16")
    smem_bytes = machine.get("smem_per_sm_bytes", 0)

    if head_dim <= 0 or head_dim % 8 != 0 or head_dim > 256:
        return "MATH_FALLBACK"

    if (cc_major, cc_minor) >= (9, 0) and dtype in ("float16", "bfloat16", "float8") and smem_bytes >= 98304:
        return "FA3_HOPPER"

    if (cc_major, cc_minor) >= (8, 0) and dtype in ("float16", "bfloat16") and smem_bytes >= 49152:
        return "FA2_CUDA"

    if (cc_major, cc_minor) >= (7, 5) and dtype in ("float16", "bfloat16") and head_dim <= 128:
        return "FA2_TRITON"

    return "MATH_FALLBACK"


def ref_explain_failure(backend, machine, input_spec):
    cc_major, cc_minor = machine.get("compute_capability", (0, 0))
    head_dim = input_spec.get("head_dim", 0)
    dtype = input_spec.get("dtype", "float16")
    smem_bytes = machine.get("smem_per_sm_bytes", 0)

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

    return "|".join(reasons) if reasons else "NO_FAILURE"


def ref_measure_cost(target_backend, fallback_backend, machine, input_spec):
    batch = input_spec.get("batch_size", 1)
    seq_len = input_spec.get("seq_len", 2048)
    num_heads = input_spec.get("num_heads", 32)
    head_dim = input_spec.get("head_dim", 128)
    dtype = input_spec.get("dtype", "float16")

    bytes_per_elem = 1 if dtype == "float8" else 2
    total_flops = 4.0 * batch * num_heads * (seq_len ** 2) * head_dim

    eff_map = {
        "FA3_HOPPER": 0.75,
        "FA2_CUDA": 0.60,
        "FA2_TRITON": 0.50,
        "MATH_FALLBACK": 0.15,
    }

    target_eff = eff_map.get(target_backend, 0.50)
    fallback_eff = eff_map.get(fallback_backend, 0.15)

    sm_count = machine.get("sm_count", 108)
    clock_ghz = machine.get("clock_ghz", 1.5)
    cc = machine.get("compute_capability", (8, 0))

    if cc >= (9, 0):
        base_flops = 1000.0 if dtype in ("float16", "bfloat16", "float8") else 200.0
    elif cc >= (8, 0):
        base_flops = 300.0 if dtype in ("float16", "bfloat16") else 100.0
    else:
        base_flops = 130.0 if dtype in ("float16", "bfloat16") else 50.0

    peak_tflops = (sm_count / 108.0) * (clock_ghz / 1.5) * base_flops
    peak_flops = peak_tflops * 1e12

    target_time_sec = total_flops / (peak_flops * target_eff)
    fallback_time_sec = total_flops / (peak_flops * fallback_eff)

    qkv_bytes = 3 * batch * seq_len * num_heads * head_dim * bytes_per_elem
    out_bytes = batch * seq_len * num_heads * head_dim * bytes_per_elem

    target_mem_bytes = qkv_bytes + out_bytes
    if fallback_backend == "MATH_FALLBACK":
        attn_matrix_bytes = batch * num_heads * seq_len * seq_len * 4
        fallback_mem_bytes = target_mem_bytes + attn_matrix_bytes
    else:
        fallback_mem_bytes = target_mem_bytes

    return {
        "target_backend": target_backend,
        "fallback_backend": fallback_backend,
        "target_latency_ms": target_time_sec * 1000.0,
        "fallback_latency_ms": fallback_time_sec * 1000.0,
        "latency_penalty_ratio": fallback_time_sec / target_time_sec,
        "memory_overhead_bytes": fallback_mem_bytes - target_mem_bytes,
        "total_flops": total_flops,
    }
