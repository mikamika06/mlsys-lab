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
