class MemoryLimiter:
    def __init__(self, max_memory_bytes):
        self.max_memory_bytes = max_memory_bytes

    def check_usage(self, current_bytes):
        return current_bytes <= self.max_memory_bytes
