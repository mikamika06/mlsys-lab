class BucketManager:
    def __init__(self, max_size_bytes):
        self.max_size_bytes = max_size_bytes
        self.current_bucket = []
        self.current_size = 0
        self.buckets = []

    def add(self, param_name, size_bytes):
        if self.current_size + size_bytes > self.max_size_bytes and self.current_bucket:
            self.buckets.append(self.current_bucket)
            self.current_bucket = []
            self.current_size = 0
        self.current_bucket.append(param_name)
        self.current_size += size_bytes

    def flush(self):
        if self.current_bucket:
            self.buckets.append(self.current_bucket)
            self.current_bucket = []
            self.current_size = 0
        return self.buckets
