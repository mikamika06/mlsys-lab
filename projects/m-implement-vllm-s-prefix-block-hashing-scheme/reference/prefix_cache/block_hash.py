import hashlib


def hash_block(token_ids, parent_hash=None):
    hasher = hashlib.sha256()
    if parent_hash is not None:
        hasher.update(str(parent_hash).encode("utf-8"))
    else:
        hasher.update(b"ROOT")
    for tid in token_ids:
        hasher.update(int(tid).to_bytes(8, byteorder="big", signed=True))
    return hasher.hexdigest()


def compute_prefix_hashes(token_ids, block_size):
    num_blocks = len(token_ids) // block_size
    hashes = []
    parent = None
    for i in range(num_blocks):
        block_tokens = token_ids[i * block_size : (i + 1) * block_size]
        h = hash_block(block_tokens, parent)
        hashes.append(h)
        parent = h
    return hashes
