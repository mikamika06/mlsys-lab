class Prefetcher:
    def __init__(self):
        raise NotImplementedError

    def should_prefetch(self, session_id: str, history: list) -> bool:
        raise NotImplementedError
