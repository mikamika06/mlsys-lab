def measure_peak_rss(base_memory, is_qlora):
    if is_qlora:
        return base_memory * 0.65
    return base_memory * 1.25
