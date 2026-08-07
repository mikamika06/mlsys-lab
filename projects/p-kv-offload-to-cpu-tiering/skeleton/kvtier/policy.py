class EvictionPolicy:
    def __init__(self, capacity: int):
        raise NotImplementedError

    def select_victim(self, active_sessions: dict) -> str:
        raise NotImplementedError
