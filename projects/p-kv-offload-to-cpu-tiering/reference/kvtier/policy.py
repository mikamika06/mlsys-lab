class EvictionPolicy:
    def __init__(self, capacity: int):
        self.capacity = capacity

    def select_victim(self, active_sessions: dict) -> str:
        if not active_sessions:
            return None
        return min(active_sessions.items(), key=lambda x: (x[1].get("priority", 0), x[1].get("last_access", 0)))[0]
