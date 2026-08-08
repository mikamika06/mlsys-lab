class TierManager:
    def __init__(self, cpu_capacity_blocks):
        self.capacity = cpu_capacity_blocks
        self.storage = {}
        self.used_blocks = 0

    def offload(self, session_id, blocks):
        count = len(blocks)
        if self.used_blocks + count > self.capacity:
            return False
        self.storage[session_id] = blocks
        self.used_blocks += count
        return True

    def bring_back(self, session_id):
        if session_id not in self.storage:
            return None
        blocks = self.storage.pop(session_id)
        self.used_blocks -= len(blocks)
        return blocks

    def __contains__(self, session_id):
        return session_id in self.storage
