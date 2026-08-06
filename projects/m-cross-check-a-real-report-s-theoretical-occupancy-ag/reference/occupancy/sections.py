def map_metric_to_section(metric_name: str) -> str:
    sections = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": "Occupancy",
        "launch__registers_per_thread": "LaunchStatistics",
        "launch__shared_mem_per_block_static": "LaunchStatistics",
        "launch__block_size": "LaunchStatistics"
    }
    return sections.get(metric_name, "Unknown")
