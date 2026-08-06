from prefix_cache.block_hash import compute_prefix_hashes


class PrefixCacheEngine:
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
                if not self.free_blocks:
                    raise RuntimeError("Out of memory blocks")
                blk_id = self.free_blocks.pop(0)
                self.hash_to_block_id[h] = blk_id
                cached_blocks.append(blk_id)

        return cached_blocks, hit_count * self.block_size

    def simulate_request(self, token_ids, computation_time_per_token=0.001):
        blocks, cached_tokens = self.allocate_or_get(token_ids)
        tokens_to_compute = len(token_ids) - cached_tokens
        ttft_ms = tokens_to_compute * computation_time_per_token * 1000.0
        return cached_tokens, ttft_ms
