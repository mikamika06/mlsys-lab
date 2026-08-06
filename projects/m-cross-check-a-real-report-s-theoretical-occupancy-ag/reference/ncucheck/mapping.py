SECTION_MAP = {
    "smsp__occupancy_theoretical.pct": "SpeedOfLight",
    "launch__registers_per_thread": "LaunchStats",
    "l1tex__t_bytes_pipe_l1tex_mem_op_read.sum": "MemoryWorkloadAnalysis",
    "smsp__sass_thread_inst_executed_op_dadd_predicated.sum": "ComputeWorkloadAnalysis",
}

def map_metric_to_section(metric_name):
    return SECTION_MAP.get(metric_name, "Unknown")
