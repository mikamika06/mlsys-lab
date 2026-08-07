class ExpertCache:
    def __init__(self, capacity_bytes):
        raise NotImplementedError

    def access(self, expert_id):
        raise NotImplementedError
