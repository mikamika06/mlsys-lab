def compute_swap_cost(num_blocks, block_bytes, pcie_bandwidth_gbps, roundtrip=True):
    """Compute transferred bytes and PCIe time cost for swapping KV cache blocks."""
    direction_factor = 2 if roundtrip else 1
    bytes_moved = int(num_blocks * block_bytes * direction_factor)
    bandwidth_bytes_per_sec = pcie_bandwidth_gbps * 1e9
    time_seconds = bytes_moved / bandwidth_bytes_per_sec
    return {"bytes_moved": bytes_moved, "time_seconds": time_seconds}
