import hashlib

def blocks_collide(req1: dict, req2: dict) -> bool:
    """
    Compute the block‑0 hash for each request and compare.
    The hash is SHA256(token_bytes + salt_bytes), first 8 bytes used.
    """
    def _hash(req):
        token_bytes = req["token"].encode("utf-8")
        salt_bytes = req["cache_salt"].to_bytes(4, byteorder="big", signed=True)
        return hashlib.sha256(token_bytes + salt_bytes).digest()[:8]

    return _hash(req1) == _hash(req2)
