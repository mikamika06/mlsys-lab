def measure_cold_start_cost(model_size_mb, download_bandwidth_mbps, init_overhead_sec):
    transfer_time = model_size_mb / download_bandwidth_mbps
    total_time = transfer_time + init_overhead_sec
    return {"transfer_time": transfer_time, "total_time": total_time}
