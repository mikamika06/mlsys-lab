class TieredStorage:
    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        raise NotImplementedError

    def access(self, session_id: str) -> str:
        raise NotImplementedError

    def evict_to_cpu(self, session_id: str) -> bool:
        raise NotImplementedError

    def bring_to_gpu(self, session_id: str) -> bool:
        raise NotImplementedError
