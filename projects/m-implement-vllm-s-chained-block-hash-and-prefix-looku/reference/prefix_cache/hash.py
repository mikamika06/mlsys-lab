import hashlib


def compute_block_hash(token_ids, parent_hash=None):
    h = hashlib.sha256()
    if parent_hash is not None:
        if isinstance(parent_hash, str):
            h.update(parent_hash.encode("utf-8"))
        elif isinstance(parent_hash, bytes):
            h.update(parent_hash)
    for tid in token_ids:
        h.update(int(tid).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def build_prefix_hash_chain(token_ids, block_size):
    num_blocks = len(token_ids) // block_size
    hashes = []
    parent_hash = None
    for i in range(num_blocks):
        block_tokens = token_ids[i * block_size : (i + 1) * block_size]
        curr_hash = compute_block_hash(block_tokens, parent_hash)
        hashes.append(curr_hash)
        parent_hash = curr_hash
    return hashes
