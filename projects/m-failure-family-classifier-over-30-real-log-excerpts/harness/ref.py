FAMILIES = [
    "CUDA_OOM", "NCCL_TIMEOUT", "KV_CACHE_EXHAUSTION",
    "TOKENIZER_MISMATCH", "SHM_CORRUPTION"
]

EXCERPTS = [
    (f"Error executing kernel on GPU 0: CUDA out of memory. Tried to allocate {i}GB.", "CUDA_OOM")
    if i % 5 == 0 else
    (f"NCCL WARN: transport failure / connection timeout in rank {i}", "NCCL_TIMEOUT")
    if i % 5 == 1 else
    (f"ValueError: PagedAttention block table allocation failed, request {i} dropped.", "KV_CACHE_EXHAUSTION")
    if i % 5 == 2 else
    (f"RuntimeError: Tokenizer vocab size mismatch between weights and config {i}.", "TOKENIZER_MISMATCH")
    for i in range(30)
]

FIXES = {
    "CUDA_OOM": {"env": "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512", "arg": "--gpu-memory-utilization 0.90"},
    "NCCL_TIMEOUT": {"env": "NCCL_BLOCKING_WAIT=1", "arg": "--distributed-executor-backend ray"},
    "KV_CACHE_EXHAUSTION": {"env": "VLLM_ENGINE_ITERATION_TIMEOUT_S=60", "arg": "--max-model-len 4096"},
    "TOKENIZER_MISMATCH": {"env": "TOKENIZERS_PARALLELISM=false", "arg": "--trust-remote-code"},
    "SHM_CORRUPTION": {"env": "TORCH_NCCL_AVOID_RECORD_STREAMS=1", "arg": "--disable-custom-all-reduce"}
}

def classify_excerpts(excerpts):
    out = []
    for text, _ in excerpts:
        if "CUDA out of memory" in text:
            out.append("CUDA_OOM")
        elif "NCCL WARN" in text:
            out.append("NCCL_TIMEOUT")
        elif "PagedAttention" in text:
            out.append("KV_CACHE_EXHAUSTION")
        elif "Tokenizer" in text:
            out.append("TOKENIZER_MISMATCH")
        else:
            out.append("SHM_CORRUPTION")
    return out

def get_minimal_fix(family):
    return FIXES.get(family, {"env": "", "arg": ""})

def reorder_logs(logs):
    return sorted(logs, key=lambda x: (x.get("timestamp", 0), x.get("rank", 0)))
