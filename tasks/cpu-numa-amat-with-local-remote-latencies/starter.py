def compute_numa_amat(
    l3_hit_time_ns: float,
    l3_miss_rate: float,
    local_dram_latency_ns: float,
    remote_base_latency_ns: float,
    num_remote_hops: int,
    per_hop_latency_ns: float,
    local_dram_fraction: float,
) -> float:
    """Return the modeled Average Memory Access Time (ns) for a NUMA system."""
    raise NotImplementedError('your code here')
