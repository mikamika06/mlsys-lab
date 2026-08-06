import numpy as np


class ColdCacheProtocol:
    """Manages cache state, memory allocations, and cold verification."""

    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.generation = 0
        self.kv_cache = {}
        self.host_page_cache = set()
        self.allocator_resets = 0

    def invalidate_host_cache(self) -> None:
        self.host_page_cache.clear()

    def reset_gpu_allocator(self) -> int:
        self.generation += 1
        self.kv_cache.clear()
        self.allocator_resets += 1
        return self.generation

    def execute_request(self, prompt_tokens: list[int]) -> dict:
        key = tuple(prompt_tokens)
        hit = key in self.kv_cache or key in self.host_page_cache
        self.kv_cache[key] = np.zeros(len(prompt_tokens), dtype=np.float32)
        self.host_page_cache.add(key)
        return {
            "hit": hit,
            "generation": self.generation,
            "tokens_length": len(prompt_tokens)
        }
