class GradientBucket:
    def __init__(self, capacity_bytes):
        raise NotImplementedError

    def add(self, tensor):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError
