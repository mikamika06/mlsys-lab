class GradientBucket:
    def __init__(self, bucket_size_bytes: int):
        self.bucket_size_bytes = bucket_size_bytes
        self.current_bytes = 0
        self.tensors = []
        self.flushed = False

    def append(self, tensor) -> bool:
        nbytes = tensor.nbytes if hasattr(tensor, "nbytes") else len(tensor) * 4
        if self.current_bytes + nbytes > self.bucket_size_bytes and len(self.tensors) > 0:
            return False
        self.tensors.append(tensor)
        self.current_bytes += nbytes
        return True

    def flush(self) -> None:
        self.flushed = True
        self.tensors.clear()
        self.current_bytes = 0
