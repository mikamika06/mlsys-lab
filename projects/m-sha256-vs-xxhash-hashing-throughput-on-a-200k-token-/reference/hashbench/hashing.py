import hashlib
import time
import xxhash


def hash_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_xxhash(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()


def measure_throughput(data: bytes, iterations: int = 5) -> dict:
    t0 = time.perf_counter()
    for _ in range(iterations):
        hash_sha256(data)
    t_sha = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        hash_xxhash(data)
    t_xx = time.perf_counter() - t0

    bytes_total = len(data) * iterations
    mb = bytes_total / (1024 * 1024)

    throughput_sha = mb / max(t_sha, 1e-9)
    throughput_xx = mb / max(t_xx, 1e-9)

    return {
        "sha256_mb_s": throughput_sha,
        "xxhash_mb_s": throughput_xx,
        "ratio": throughput_xx / max(throughput_sha, 1e-9),
    }
