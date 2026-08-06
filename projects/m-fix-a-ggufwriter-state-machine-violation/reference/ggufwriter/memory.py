def estimate_peak_memory(file_size, use_memmap=True):
    if use_memmap:
        return 4096 + (1024 * 64)
    return file_size + 4096
