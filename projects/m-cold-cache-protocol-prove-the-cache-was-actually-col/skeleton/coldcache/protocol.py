class ColdCacheProtocol:
    """Manages cache state, memory allocations, and cold verification."""

    def __init__(self, memory_size: int):
        raise NotImplementedError

    def invalidate_host_cache(self) -> None:
        raise NotImplementedError

    def reset_gpu_allocator(self) -> int:
        raise NotImplementedError

    def execute_request(self, prompt_tokens: list[int]) -> dict:
        raise NotImplementedError
