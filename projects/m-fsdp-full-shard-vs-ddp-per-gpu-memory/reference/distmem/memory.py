def compute_memory_comparison(
    params, bytes_per_param, optimizer_bytes_per_param, world_size
):
    param_mem = params * bytes_per_param
    grad_mem = params * bytes_per_param
    opt_mem = params * optimizer_bytes_per_param
    ddp_mem = param_mem + grad_mem + opt_mem

    sharded_param = (params * bytes_per_param) / world_size
    sharded_grad = (params * bytes_per_param) / world_size
    sharded_opt = (params * optimizer_bytes_per_param) / world_size
    unsharded_current_layer = (params / 32) * bytes_per_param
    fsdp_mem = (
        sharded_param + sharded_grad + sharded_opt + unsharded_current_layer
    )
    return ddp_mem, fsdp_mem
