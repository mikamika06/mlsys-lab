MEMORY_CASES = [
    {
        "params": 1000000000,
        "bytes_per_param": 2,
        "optimizer_bytes_per_param": 12,
        "world_size": 8,
    },
    {
        "params": 7000000000,
        "bytes_per_param": 2,
        "optimizer_bytes_per_param": 16,
        "world_size": 4,
    },
    {
        "params": 13000000000,
        "bytes_per_param": 4,
        "optimizer_bytes_per_param": 12,
        "world_size": 8,
    },
]


def ref_ddp_memory(params, bytes_per_param, optimizer_bytes_per_param):
    param_mem = params * bytes_per_param
    grad_mem = params * bytes_per_param
    opt_mem = params * optimizer_bytes_per_param
    return param_mem + grad_mem + opt_mem


def ref_fsdp_memory(
    params, bytes_per_param, optimizer_bytes_per_param, world_size
):
    sharded_param = (params * bytes_per_param) / world_size
    sharded_grad = (params * bytes_per_param) / world_size
    sharded_opt = (params * optimizer_bytes_per_param) / world_size
    unsharded_current_layer = (params / 32) * bytes_per_param
    return (
        sharded_param + sharded_grad + sharded_opt + unsharded_current_layer
    )


TRACE_CASES = [
    {
        "events": ["all_gather", "compute", "reduce_scatter"],
        "expected": "fsdp_fully_sharded_pipelined",
    },
    {
        "events": ["all_reduce", "compute", "bucket_sync"],
        "expected": "ddp_bucketed_allreduce",
    },
    {
        "events": ["wait_stream", "idle", "all_gather_stall"],
        "expected": "fsdp_communication_stall",
    },
]


def ref_classify_trace(events):
    if "all_gather" in events and "reduce_scatter" in events:
        return "fsdp_fully_sharded_pipelined"
    elif "all_reduce" in events:
        return "ddp_bucketed_allreduce"
    return "fsdp_communication_stall"


OPTIMIZER_CASES = [
    {"hooks": True, "splitting": True, "expected_breaks": 3},
    {"hooks": False, "splitting": False, "expected_breaks": 0},
    {"hooks": True, "splitting": False, "expected_breaks": 1},
]


def ref_count_graph_breaks(hooks, splitting):
    breaks = 0
    if hooks:
        breaks += 1
    if splitting:
        breaks += 2
    return breaks
