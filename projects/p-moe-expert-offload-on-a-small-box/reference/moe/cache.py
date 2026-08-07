class ExpertCache:
    def __init__(self, capacity_bytes):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.cache = {}
        self.order = []

    def access(self, expert_id, size_bytes=1000):
        if expert_id in self.cache:
            self.order.remove(expert_id)
            self.order.append(expert_id)
            return True

        while self.current_bytes + size_bytes > self.capacity_bytes and self.order:
            oldest = self.order.pop(0)
            self.current_bytes -= self.cache.pop(oldest, 0)

        if self.current_bytes + size_bytes <= self.capacity_bytes:
            self.cache[expert_id] = size_bytes
            self.order.append(expert_id)
            self.current_bytes += size_bytes
            return False
        return False
