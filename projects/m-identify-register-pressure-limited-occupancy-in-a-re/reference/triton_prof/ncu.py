def analyze_occupancy(profile_data):
    """Analyze Triton NCU profile data to determine occupancy limiting factor."""
    regs = profile_data["regs_per_thread"]
    threads = profile_data["threads_per_block"]
    warp_size = profile_data.get("warp_size", 32)
    warps = (threads + warp_size - 1) // warp_size
    reg_limit_blocks = 65536 // (regs * warps * warp_size) if regs > 0 else 32
    smem_limit = 49152 // max(1, profile_data["smem_bytes"])
    thread_limit = 2048 // threads
    limits = {"register": reg_limit_blocks, "smem": smem_limit, "threads": thread_limit}
    bottleneck = min(limits, key=limits.get)
    return {"bottleneck": bottleneck, "max_active_blocks": limits[bottleneck]}
