class KeepAliveController:
    def __init__(self, memory_cap_mb):
        raise NotImplementedError

    def update(self, active_requests):
        raise NotImplementedError
