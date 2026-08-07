from typing import Any, Dict, List


def calculate_kv_bytes(
    config: Dict[str, Any],
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """Calculate total and per-layer KV cache memory consumption in bytes."""
    raise NotImplementedError


def calculate_memory_savings(
    config: Dict[str, Any],
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """Calculate fraction of memory saved compared to full context attention."""
    raise NotImplementedError
