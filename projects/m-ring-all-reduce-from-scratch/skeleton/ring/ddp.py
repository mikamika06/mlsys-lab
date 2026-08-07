class BucketManager:
    def __init__(self, max_size_bytes):
        raise NotImplementedError

    def add(self, param_name, size_bytes):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError
