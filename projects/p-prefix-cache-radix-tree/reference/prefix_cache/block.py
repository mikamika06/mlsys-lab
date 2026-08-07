import hashlib


class BlockHasher:
    """Computes individual and chained hashes for token blocks."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size

    def hash_block(self, tokens: list[int]) -> int:
        hasher = hashlib.sha256()
        for token in tokens:
            hasher.update(token.to_bytes(8, byteorder="big", signed=True))
        return int.from_bytes(hasher.digest()[:8], byteorder="big")

    def compute_chain_hashes(self, tokens: list[int], tenant_id: str = "") -> list[int]:
        chain_hashes = []
        prev_hash = int.from_bytes(hashlib.sha256(tenant_id.encode("utf-8")).digest()[:8], byteorder="big")

        num_blocks = len(tokens) // self.block_size
        for i in range(num_blocks):
            block_tokens = tokens[i * self.block_size : (i + 1) * self.block_size]
            b_hash = self.hash_block(block_tokens)

            combined = hashlib.sha256()
            combined.update(prev_hash.to_bytes(8, byteorder="big"))
            combined.update(b_hash.to_bytes(8, byteorder="big"))
            prev_hash = int.from_bytes(combined.digest()[:8], byteorder="big")
            chain_hashes.append(prev_hash)

        return chain_hashes
