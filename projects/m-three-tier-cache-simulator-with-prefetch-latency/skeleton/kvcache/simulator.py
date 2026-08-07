class ThreeTierCache:
    """Three-tier cache simulator with prefetch latency."""

    def __init__(self, vram_cap: int, dram_cap: int, nvme_cap: int,
                 vram_lat: float = 1.0, dram_lat: float = 10.0, nvme_lat: float = 100.0,
                 dram_bw_gbps: float = 50.0, nvme_bw_gbps: float = 10.0):
        raise NotImplementedError

    def access(self, key: str, size_bytes: int, current_time: float, prefetch_keys: list[str] = None) -> dict:
        raise NotImplementedError
