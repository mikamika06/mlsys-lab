import hashlib

def compute_block_hash(token_block, salt):
    h = hashlib.sha256()
    h.update(salt.encode())
    h.update(bytes(token_block))
    return h.digest()

def verify_tenant_isolation(blocks_a, blocks_b, salt_a, salt_b):
    hashes_a = {compute_block_hash(b, salt_a) for b in blocks_a}
    hashes_b = {compute_block_hash(b, salt_b) for b in blocks_b}
    return len(hashes_a.intersection(hashes_b)) == 0
