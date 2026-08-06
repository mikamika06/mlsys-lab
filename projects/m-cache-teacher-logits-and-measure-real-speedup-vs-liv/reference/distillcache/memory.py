def compute_full_vocab_footprint(num_samples: int, seq_len: int, vocab_size: int, dtype_bytes: int = 4) -> int:
    """Calculate memory footprint in bytes for full vocabulary teacher logits."""
    return int(num_samples * seq_len * vocab_size * dtype_bytes)


def compute_topk_footprint(num_samples: int, seq_len: int, top_k: int, logit_dtype_bytes: int = 2, index_dtype_bytes: int = 2) -> int:
    """Calculate memory footprint in bytes for top-k compressed teacher logits and indices."""
    return int(num_samples * seq_len * top_k * (logit_dtype_bytes + index_dtype_bytes))


def max_samples_within_budget(ram_budget_bytes: int, seq_len: int, vocab_size: int, dtype_bytes: int = 4) -> int:
    """Calculate maximum number of dataset samples fitting within RAM budget."""
    bytes_per_sample = seq_len * vocab_size * dtype_bytes
    if bytes_per_sample <= 0:
        return 0
    return int(ram_budget_bytes // bytes_per_sample)
