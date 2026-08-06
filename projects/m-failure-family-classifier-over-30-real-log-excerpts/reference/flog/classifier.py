def classify_failure(excerpt):
    e = excerpt.lower()
    if "nccl" in e or "timeout" in e:
        return "nccl_timeout"
    if "oom" in e or "out of memory" in e or "cuda out of memory" in e:
        return "cuda_oom"
    if "flash" in e or "attn" in e or "flashinfer" in e:
        return "kernel_assertion"
    if "kv" in e and "cache" in e:
        return "kv_cache_overflow"
    return "unknown_fault"
