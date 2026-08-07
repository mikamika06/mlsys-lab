import hashlib


def compute_block_hash(tenant_id, block_tokens, salt="", parent_hash=""):
    """Compute salted hash for a single KV cache block."""
    hasher = hashlib.sha256()
    hasher.update(str(tenant_id).encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(salt).encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(parent_hash).encode("utf-8"))
    hasher.update(b":")
    for tok in block_tokens:
        hasher.update(int(tok).to_bytes(4, byteorder="big", signed=True))
        hasher.update(b",")
    return hasher.hexdigest()


def build_prefix_keys(tenant_id, tokens, block_size, salt=""):
    """Build sequence of block cache keys for a given token sequence."""
    if block_size <= 0:
        return []
    keys = []
    parent_hash = ""
    num_blocks = len(tokens) // block_size
    for i in range(num_blocks):
        block_tokens = tokens[i * block_size : (i + 1) * block_size]
        h = compute_block_hash(tenant_id, block_tokens, salt=salt, parent_hash=parent_hash)
        keys.append(h)
        parent_hash = h
    return keys
