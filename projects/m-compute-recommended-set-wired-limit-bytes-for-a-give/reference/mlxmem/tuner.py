from mlxmem.limits import (
    compute_recommended_cache_limit,
    compute_recommended_wired_limit,
)


class MemoryTuner:
    """Tune cache and wired limits to maximize throughput without OOM."""

    def __init__(self, hw_memsize_bytes: int):
        self.hw_memsize = hw_memsize_bytes
        self.wired_limit = compute_recommended_wired_limit(hw_memsize_bytes)
        self.cache_limit = 64 * 1024 * 1024
        self.active_model_bytes = 0
        self.kv_bytes = 0

    def configure(self, active_model_bytes: int) -> dict:
        self.active_model_bytes = active_model_bytes
        self.cache_limit = compute_recommended_cache_limit(self.hw_memsize, active_model_bytes)
        return {
            "wired_limit_bytes": self.wired_limit,
            "cache_limit_bytes": self.cache_limit,
            "status": "configured"
        }

    def step_generation(self, token_idx: int, kv_delta_bytes: int) -> dict:
        self.kv_bytes += kv_delta_bytes
        total_retained = self.active_model_bytes + self.kv_bytes
        headroom = self.wired_limit - total_retained

        if headroom <= 0:
            self.cache_limit = 32 * 1024 * 1024
            action = "throttled_cache"
        elif total_retained + self.cache_limit > self.wired_limit:
            self.cache_limit = max(32 * 1024 * 1024, int(headroom * 0.25))
            action = "adjusted_cache"
        else:
            action = "nominal"

        return {
            "token_idx": token_idx,
            "wired_limit_bytes": self.wired_limit,
            "cache_limit_bytes": self.cache_limit,
            "total_retained_bytes": total_retained,
            "action": action
        }
