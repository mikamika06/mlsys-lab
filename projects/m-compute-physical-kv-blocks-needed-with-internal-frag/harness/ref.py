import math
import re


def compute_physical_blocks_needed(seq_lens: list[int], block_size: int) -> dict[str, float]:
    total_blocks = 0
    total_used_tokens = 0
    for l in seq_lens:
        if l > 0:
            blocks = math.ceil(l / block_size)
            total_blocks += blocks
            total_used_tokens += l
    total_capacity = total_blocks * block_size
    unused_tokens = total_capacity - total_used_tokens
    frag_ratio = unused_tokens / total_capacity if total_capacity > 0 else 0.0
    return {
        "total_blocks": float(total_blocks),
        "total_capacity": float(total_capacity),
        "total_used_tokens": float(total_used_tokens),
        "fragmentation_ratio": float(frag_ratio),
    }


def parse_vllm_startup_and_compute_capacity(
    log_text: str,
    seq_len: int,
    block_size: int,
    bytes_per_token: int
) -> dict[str, float]:
    match = re.search(r"# GPU KV cache memory:\s*([\d\.]+)\050?\s*GiB", log_text, re.IGNORECASE)
    if not match:
        match = re.search(r"GPU KV cache size:\s*([\d\.]+)\s*GiB", log_text, re.IGNORECASE)
    if not match:
        raise ValueError("Could not parse GPU KV cache memory from startup log")

    kv_cache_gib = float(match.group(1))
    kv_cache_bytes = kv_cache_gib * (1024 ** 3)
    bytes_per_block = block_size * bytes_per_token
    total_blocks = int(kv_cache_bytes // bytes_per_block)
    blocks_per_seq = math.ceil(seq_len / block_size)
    max_concurrent_seqs = total_blocks // blocks_per_seq if blocks_per_seq > 0 else 0

    return {
        "kv_cache_gib": float(kv_cache_gib),
        "total_blocks": float(total_blocks),
        "blocks_per_seq": float(blocks_per_seq),
        "max_concurrent_seqs": float(max_concurrent_seqs),
    }


SAMPLE_LOGS = [
    """INFO 08-06 14:02:00 config.py:1020] Memory profiling results:
INFO 08-06 14:02:00 config.py:1020] # GPU KV cache memory: 24.50 GiB
INFO 08-06 14:02:00 config.py:1020] # CPU KV cache memory: 4.00 GiB""",
    """[2026-08-06 10:00:00] INFO: GPU KV Cache size: 12.25 GiB allocated.""",
    """INFO: Initialized vLLM engine.
# GPU KV cache memory: 80.00 GiB available for blocks."""
]
