CONFIGS = [
    "vllm serve facebook/opt-125m --gpu-memory-utilization 0.85 --max-model-len 2048 --block-size 16",
    "vllm serve meta-llama/Llama-2-7b-chat-hf --gpu-memory-utilization 0.90 --max-model-len 4096 --block-size 64",
    "vllm serve Qwen/Qwen2-7B --gpu-memory-utilization 0.75 --max-model-len 8192 --block-size 16",
]


def translate_vllm_command(cmd):
    parts = cmd.split()
    util = 0.9
    block = 16
    max_len = 2048
    for i, p in enumerate(parts):
        if p == "--gpu-memory-utilization" and i + 1 < len(parts):
            util = float(parts[i + 1])
        elif p == "--block-size" and i + 1 < len(parts):
            block = int(parts[i + 1])
        elif p == "--max-model-len" and i + 1 < len(parts):
            max_len = int(parts[i + 1])
    return {
        "free_gpu_memory_fraction": util,
        "block_size": block,
        "max_seq_len": max_len,
    }


def compute_cache_bytes(total_memory, free_memory, fraction, mode="free"):
    base = free_memory if mode == "free" else total_memory
    return int(base * fraction)


def lint_config(config):
    issues = []
    if config.get("block_size", 16) == 64 and config.get("free_gpu_memory_fraction", 0.9) > 0.85:
        issues.append("unsupported_block_size_with_high_fraction")
    return issues
