def block_hash(tokens: list[int]) -> int:
    """Compute a deterministic hash for a sequence of token IDs."""
    h = 14695981039346656037
    for t in tokens:
        h ^= (t & 0xFFFFFFFF)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def tokenize_into_blocks(tokens: list[int], block_size: int) -> list[int]:
    """Split token IDs into fixed-size block hashes."""
    if not tokens or block_size <= 0:
        return []
    blocks = []
    for i in range(0, len(tokens), block_size):
        chunk = tokens[i:i + block_size]
        blocks.append(block_hash(chunk))
    return blocks


def compute_prefix_match(req_blocks: list[int], worker_blocks: list[int]) -> int:
    """Compute contiguous prefix block match count from block 0."""
    match = 0
    min_len = min(len(req_blocks), len(worker_blocks))
    for i in range(min_len):
        if req_blocks[i] == worker_blocks[i]:
            match += 1
        else:
            break
    return match


class PrefixRouter:
    """Prefix-hash router tracking worker block caches."""

    def __init__(self, num_workers: int, max_blocks_per_worker: int, block_size: int):
        self.num_workers = num_workers
        self.max_blocks_per_worker = max_blocks_per_worker
        self.block_size = block_size
        self.worker_caches: list[list[int]] = [[] for _ in range(num_workers)]

    def get_worker_blocks(self, worker_id: int) -> list[int]:
        return list(self.worker_caches[worker_id])

    def route(self, tokens: list[int]) -> tuple[int, int]:
        req_blocks = tokenize_into_blocks(tokens, self.block_size)
        best_worker = 0
        best_match = -1
        for w in range(self.num_workers):
            match = compute_prefix_match(req_blocks, self.worker_caches[w])
            if match > best_match:
                best_match = match
                best_worker = w
        return best_worker, max(0, best_match)

    def update_cache(self, worker_id: int, tokens: list[int]):
        req_blocks = tokenize_into_blocks(tokens, self.block_size)
        if len(req_blocks) > self.max_blocks_per_worker:
            self.worker_caches[worker_id] = req_blocks[:self.max_blocks_per_worker]
        else:
            self.worker_caches[worker_id] = list(req_blocks)
