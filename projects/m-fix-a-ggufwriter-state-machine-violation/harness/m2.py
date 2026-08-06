import ref

def check(workdir):
    from ggufwriter.memory import estimate_peak_memory
    out = {"memory_ratio_ok": 0.0}
    file_size = 1024 * 1024 * 100
    mem_map = estimate_peak_memory(file_size, use_memmap=True)
    mem_full = estimate_peak_memory(file_size, use_memmap=False)
    if mem_map < mem_full and mem_map < file_size:
        out["memory_ratio_ok"] = 1.0
    else:
        out["_note"] = f"Memmap peak memory {mem_map} not sufficiently smaller than full read {mem_full}"
    return out
