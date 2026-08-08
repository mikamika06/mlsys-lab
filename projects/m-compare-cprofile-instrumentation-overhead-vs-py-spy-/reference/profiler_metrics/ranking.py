def rank_profiler_flags(flag_measurements):
    sorted_flags = sorted(flag_measurements.items(), key=lambda x: x[1])
    return [flag for flag, overhead in sorted_flags]
