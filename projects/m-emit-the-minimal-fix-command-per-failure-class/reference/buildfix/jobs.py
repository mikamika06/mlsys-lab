def compute_max_jobs(available_ram_gb, core_count, gb_per_job=4):
    by_ram = max(1, int(available_ram_gb // gb_per_job))
    return min(core_count, by_ram)
