def compare_mmap_latencies(model_size_bytes, disk_read_bw_bytes_sec, page_fault_overhead_sec, is_page_cache_warm):
    """Compare cold-start startup latency for mmap vs --no-mmap modes."""
    no_mmap_latency = (model_size_bytes / disk_read_bw_bytes_sec) if not is_page_cache_warm else 0.05
    
    if is_page_cache_warm:
        mmap_latency = page_fault_overhead_sec * 0.1
    else:
        mmap_latency = page_fault_overhead_sec + (model_size_bytes / disk_read_bw_bytes_sec) * 0.1
        
    faster_mode = "mmap" if mmap_latency < no_mmap_latency else "no-mmap"
    
    return {
        "mmap_latency_sec": mmap_latency,
        "no_mmap_latency_sec": no_mmap_latency,
        "recommended_mode": faster_mode
    }
