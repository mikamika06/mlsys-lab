class TieredStorage:
    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        self.gpu_cap = gpu_capacity
        self.cpu_cap = cpu_capacity
        self.gpu_set = set()
        self.cpu_set = set()

    def access(self, session_id: str) -> str:
        if session_id in self.gpu_set:
            return "gpu"
        if session_id in self.cpu_set:
            return "cpu"
        if len(self.gpu_set) < self.gpu_cap:
            self.gpu_set.add(session_id)
            return "gpu"
        return "none"

    def evict_to_cpu(self, session_id: str) -> bool:
        if session_id in self.gpu_set and len(self.cpu_set) < self.cpu_cap:
            self.gpu_set.remove(session_id)
            self.cpu_set.add(session_id)
            return True
        return False

    def bring_to_gpu(self, session_id: str) -> bool:
        if session_id in self.cpu_set and len(self.gpu_set) < self.gpu_cap:
            self.cpu_set.remove(session_id)
            self.gpu_set.add(session_id)
            return True
        return False
