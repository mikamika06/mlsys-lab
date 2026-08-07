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


class PrefixCacheManager:

    def __init__(self, block_size):
        self.block_size = block_size
        self.cached_hashes = set()

    def insert_sequence(self, token_ids):
        chain = build_prefix_hash_chain(token_ids, self.block_size)
        for h in chain:
            self.cached_hashes.add(h)
        return len(chain)

    def lookup_prefix(self, token_ids):
        chain = build_prefix_hash_chain(token_ids, self.block_size)
        matched_blocks = 0
        for h in chain:
            if h in self.cached_hashes:
                matched_blocks += 1
            else:
                break
        matched_tokens = matched_blocks * self.block_size
        return matched_blocks, matched_tokens


def compute_trace_hit_rate(requests, block_size):
    mgr = PrefixCacheManager(block_size)
    total_tokens = 0
    total_cached_tokens = 0
    for req in requests:
        n_tokens = len(req)
        total_tokens += n_tokens
        if n_tokens < block_size:
            continue
        matched_blocks, matched_tokens = mgr.lookup_prefix(req)
        total_cached_tokens += matched_tokens
        mgr.insert_sequence(req)
    if total_tokens == 0:
        return 0.0
    return total_cached_tokens / total_tokens


def optimize_prompt_layout(components, block_size):
    stable = [c for c in components if c.get("stable", False)]
    volatile = [c for c in components if not c.get("stable", False)]

    reordered = stable + volatile
    optimized_tokens = []
    for c in reordered:
        optimized_tokens.extend(c["tokens"])

    full_block_tokens = (len(optimized_tokens) // block_size) * block_size
    prefix_tokens = optimized_tokens[:full_block_tokens]

    return {
        "optimized_components": reordered,
        "prompt_tokens": optimized_tokens,
        "prefix_block_tokens": prefix_tokens,
    }


TEST_TOKEN_SEQS = [
    [101, 202, 303, 404, 505, 606, 707, 808],
    [101, 202, 303, 404, 999, 888, 777, 666],
    [999, 888, 303, 404, 505, 606, 707, 808],
    [1, 2, 3, 4, 5],
]

TEST_TRACES = [
    [
        [10, 20, 30, 40, 50, 60, 70, 80],
        [10, 20, 30, 40, 50, 60, 99, 88],
        [10, 20, 30, 40, 11, 22, 33, 44],
        [99, 88, 77, 66, 55, 44, 33, 22],
    ],
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99, 98, 97, 96],
        [1, 2, 3, 4, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88],
    ],
]

TEST_COMPONENTS = [
    [
        {
            "name": "timestamp",
            "stable": False,
            "tokens": [2026, 8, 7, 9, 21],
        },
        {
            "name": "system_prompt",
            "stable": True,
            "tokens": [100, 101, 102, 103, 104, 105, 106, 107],
        },
        {
            "name": "user_id",
            "stable": False,
            "tokens": [99999],
        },
        {
            "name": "tools_definition",
            "stable": True,
            "tokens": [200, 201, 202, 203, 204, 205, 206, 207],
        },
    ],
    [
        {"name": "session_id", "stable": False, "tokens": [77, 88]},
        {
            "name": "preamble",
            "stable": True,
            "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    ],
]
