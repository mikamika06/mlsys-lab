import hashlib

DATASET = [
    [101, 2054, 2003, 1037, 3899, 2000, 1037, 1010] * 4,
    [101, 2054, 2003, 1037, 3899, 2000, 1037, 1010] * 4 + [2054, 2003],
    [101, 2054, 2003, 1037, 9999, 2000, 1037, 1010] * 4,
    [101, 2054, 2003, 1037, 3899, 2000, 1037, 1010] * 8,
]

BLOCK_SIZE = 8
TOTAL_BLOCKS = 64


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


class ReferenceCacheEngine:
    def __init__(self, block_size, total_blocks):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.hash_to_block_id = {}
        self.free_blocks = list(range(total_blocks))

    def allocate_or_get(self, token_ids):
        hashes = compute_prefix_hashes(token_ids, self.block_size)
        cached_blocks = []
        hit_count = 0
        missed = False

        for h in hashes:
            if not missed and h in self.hash_to_block_id:
                cached_blocks.append(self.hash_to_block_id[h])
                hit_count += 1
            else:
                missed = True
                blk_id = self.free_blocks.pop(0)
                self.hash_to_block_id[h] = blk_id
                cached_blocks.append(blk_id)

        return cached_blocks, hit_count * self.block_size

    def simulate_request(self, token_ids, computation_time_per_token=0.001):
        blocks, cached_tokens = self.allocate_or_get(token_ids)
        tokens_to_compute = len(token_ids) - cached_tokens
        ttft_ms = tokens_to_compute * computation_time_per_token * 1000.0
        return cached_tokens, ttft_ms


class ReferenceMetricsTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.requests = []

    def record_request(self, total_tokens, cached_tokens, ttft_ms):
        self.total_prompt_tokens += total_tokens
        self.total_cached_tokens += cached_tokens
        self.requests.append(
            {"total": total_tokens, "cached": cached_tokens, "ttft": ttft_ms}
        )

    def get_hit_rate(self):
        if self.total_prompt_tokens == 0:
            return 0.0
        return float(self.total_cached_tokens) / float(self.total_prompt_tokens)
