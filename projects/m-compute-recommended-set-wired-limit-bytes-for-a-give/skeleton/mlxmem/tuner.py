class MemoryTuner:
    """Tune cache and wired limits to maximize throughput without OOM."""

    def __init__(self, hw_memsize_bytes: int):
        raise NotImplementedError

    def configure(self, active_model_bytes: int) -> dict:
        raise NotImplementedError

    def step_generation(self, token_idx: int, kv_delta_bytes: int) -> dict:
        raise NotImplementedError
