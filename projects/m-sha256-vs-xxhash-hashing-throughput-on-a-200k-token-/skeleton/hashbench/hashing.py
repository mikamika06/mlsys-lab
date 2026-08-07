def hash_sha256(data: bytes) -> str:
    raise NotImplementedError


def hash_xxhash(data: bytes) -> str:
    raise NotImplementedError


def measure_throughput(data: bytes, iterations: int = 5) -> dict:
    raise NotImplementedError
