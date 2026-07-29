def offload_split(total_bytes, gpu_ratio, page_size=4096):
    raw_gpu = int(total_bytes * gpu_ratio)
    gpu_bytes = (raw_gpu // page_size) * page_size
    if gpu_bytes > total_bytes:
        gpu_bytes = total_bytes
    if gpu_bytes < 0:
        gpu_bytes = 0
    cpu_bytes = total_bytes - gpu_bytes
    return {"gpu_bytes": gpu_bytes, "cpu_bytes": cpu_bytes}
