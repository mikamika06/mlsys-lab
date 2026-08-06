def classify_trace_pattern(events):
    if "all_gather" in events and "reduce_scatter" in events:
        return "fsdp_fully_sharded_pipelined"
    elif "all_reduce" in events:
        return "ddp_bucketed_allreduce"
    return "fsdp_communication_stall"
