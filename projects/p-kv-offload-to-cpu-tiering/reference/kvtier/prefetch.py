class Prefetcher:
    def __init__(self):
        self.history = {}

    def should_prefetch(self, session_id: str, history: list) -> bool:
        if len(history) < 2:
            return False
        return history[-1] - history[-2] < 5.0
