def rank_profiler_options():
    """Rank torch profiler options by added overhead."""
    overheads = {
        "with_stack": 1.85,
        "profile_memory": 1.45,
        "record_shapes": 1.15,
        "with_flops": 1.10
    }
    return sorted(overheads.keys(), key=lambda k: overheads[k], reverse=True)
