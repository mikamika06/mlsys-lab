def lint_config(config):
    issues = []
    if config.get("block_size", 16) == 64 and config.get("free_gpu_memory_fraction", 0.9) > 0.85:
        issues.append("unsupported_block_size_with_high_fraction")
    return issues
