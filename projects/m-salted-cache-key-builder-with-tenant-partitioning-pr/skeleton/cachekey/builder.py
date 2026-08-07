def compute_block_hash(tenant_id, block_tokens, salt="", parent_hash=""):
    """Compute salted hash for a single KV cache block."""
    raise NotImplementedError


def build_prefix_keys(tenant_id, tokens, block_size, salt=""):
    """Build sequence of block cache keys for a given token sequence."""
    raise NotImplementedError
