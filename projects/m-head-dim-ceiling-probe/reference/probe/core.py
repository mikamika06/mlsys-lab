def compute_head_dim_ceiling(hardware_spec):
    max_shared_mem = hardware_spec.get("max_shared_mem_per_block", 49152)
    elem_size = hardware_spec.get("elem_size_bytes", 2)
    max_threads = hardware_spec.get("max_threads_per_block", 1024)
    ceilings = []
    for hd in [32, 64, 96, 128, 160, 192, 256]:
        req_smem = hd * max_threads * elem_size * 2
        if req_smem <= max_shared_mem and hd <= 256:
            ceilings.append(hd)
    return max(ceilings) if ceilings else 64
