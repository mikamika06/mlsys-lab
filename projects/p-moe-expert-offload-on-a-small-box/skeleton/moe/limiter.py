class MemoryLimiter:
    def __init__(self, max_memory_bytes):
        raise NotImplementedError

    def check_usage(self, current_bytes):
        raise NotImplementedError
