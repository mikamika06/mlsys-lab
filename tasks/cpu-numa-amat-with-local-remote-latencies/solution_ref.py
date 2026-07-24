def compute_numa_amat(
    l3_hit_time_ns: float,
    l3_miss_rate: float,
    local_dram_latency_ns: float,
    remote_base_latency_ns: float,
    num_remote_hops: int,
    per_hop_latency_ns: float,
    local_dram_fraction: float,
) -> float:
    """Return the modeled Average Memory Access Time (ns) for a NUMA system.

    Uses the standard AMAT formula with a NUMA-weighted DRAM latency:
      AMAT = t_cache + r * (f * t_local + (1-f) * t_remote)
    where t_remote = t_base + (h-1) * t_hop.
    """
    remote_latency = remote_base_latency_ns + max(num_remote_hops - 1, 0) * per_hop_latency_ns
    dram_latency = (local_dram_fraction * local_dram_latency_ns
                    + (1.0 - local_dram_fraction) * remote_latency)
    return l3_hit_time_ns + l3_miss_rate * dram_latency
