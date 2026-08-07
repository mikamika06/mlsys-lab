from prefix_cache.hash import build_prefix_hash_chain


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
