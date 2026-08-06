import re
import math


def parse_vllm_startup_and_compute_capacity(
    log_text: str,
    seq_len: int,
    block_size: int,
    bytes_per_token: int
) -> dict[str, float]:
    """
    Parses vLLM startup log to extract available GPU KV cache memory in GiB,
    and calculates total available blocks and max concurrent sequences accounting for tail block fragmentation.
    """
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
