class GradientBucket:
    def __init__(self, bucket_size_bytes: int):
        raise NotImplementedError

    def append(self, tensor) -> bool:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError
