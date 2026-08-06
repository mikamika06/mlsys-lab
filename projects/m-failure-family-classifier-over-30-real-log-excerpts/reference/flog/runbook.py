def get_fix(family):
    mapping = {
        "nccl_timeout": {"env": {"NCCL_ASYNC_ERROR_HANDLING": "1"}, "arg": "--max-model-len 4096"},
        "cuda_oom": {"env": {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, "arg": "--gpu-memory-utilization 0.90"},
        "kernel_assertion": {"env": {}, "arg": "--disable-custom-all-reduce"},
        "kv_cache_overflow": {"env": {}, "arg": "--block-size 16"},
        "unknown_fault": {"env": {}, "arg": "--help"}
    }
    return mapping.get(family, {"env": {}, "arg": ""})
