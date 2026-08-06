from mlxmem.limits import (
    compute_recommended_cache_limit,
    compute_recommended_wired_limit,
)


class RSSTracker:
    """Track and compare RSS growth over N tokens with vs without set_cache_limit."""

    def __init__(self, hw_memsize_bytes: int):
        self.hw_memsize = hw_memsize_bytes
        self.wired_limit = compute_recommended_wired_limit(hw_memsize_bytes)

    def simulate_generation(self, num_tokens: int, token_alloc_bytes: int, use_cache_limit: bool) -> list:
        rss_samples = []
        model_bytes = int(self.hw_memsize * 0.35)
        current_rss = model_bytes
        cache_accumulator = 0
        cache_limit = compute_recommended_cache_limit(self.hw_memsize, model_bytes) if use_cache_limit else float("inf")

        for _ in range(num_tokens):
            current_rss += token_alloc_bytes
            cache_accumulator += int(token_alloc_bytes * 0.75)

            if use_cache_limit and cache_accumulator > cache_limit:
                freed = cache_accumulator - cache_limit
                cache_accumulator = cache_limit
                current_rss -= freed

            rss_samples.append(current_rss)

        return rss_samples

    def compare_rss_growth(self, num_tokens: int, token_alloc_bytes: int) -> dict:
        without_cache = self.simulate_generation(num_tokens, token_alloc_bytes, use_cache_limit=False)
        with_cache = self.simulate_generation(num_tokens, token_alloc_bytes, use_cache_limit=True)

        final_without = without_cache[-1] if without_cache else 0
        final_with = with_cache[-1] if with_cache else 0
        reduction_ratio = (final_without - final_with) / float(final_without) if final_without > 0 else 0.0

        return {
            "rss_without_cache_limit": without_cache,
            "rss_with_cache_limit": with_cache,
            "final_rss_without_bytes": final_without,
            "final_rss_with_bytes": final_with,
            "rss_reduction_ratio": reduction_ratio
        }
