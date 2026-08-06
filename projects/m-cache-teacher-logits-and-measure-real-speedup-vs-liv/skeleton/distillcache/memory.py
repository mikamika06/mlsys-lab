def compute_full_vocab_footprint(num_samples: int, seq_len: int, vocab_size: int, dtype_bytes: int = 4) -> int:
    """Calculate memory footprint in bytes for full vocabulary teacher logits."""
    raise NotImplementedError


def compute_topk_footprint(num_samples: int, seq_len: int, top_k: int, logit_dtype_bytes: int = 2, index_dtype_bytes: int = 2) -> int:
    """Calculate memory footprint in bytes for top-k compressed teacher logits and indices."""
    raise NotImplementedError


def max_samples_within_budget(ram_budget_bytes: int, seq_len: int, vocab_size: int, dtype_bytes: int = 4) -> int:
    """Calculate maximum number of dataset samples fitting within RAM budget."""
    raise NotImplementedError
