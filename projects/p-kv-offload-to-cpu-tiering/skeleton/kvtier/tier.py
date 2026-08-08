class TierManager:
    def __init__(self, cpu_capacity_blocks):
        raise NotImplementedError

    def offload(self, session_id, blocks):
        raise NotImplementedError

    def bring_back(self, session_id):
        raise NotImplementedError
