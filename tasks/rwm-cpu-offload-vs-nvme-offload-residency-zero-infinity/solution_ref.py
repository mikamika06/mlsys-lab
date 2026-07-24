def zero_residency(
    offload_optimizer,
    offload_param,
    num_params,
    param_bytes=2,
    optimizer_bytes_per_param=12,
):
    param_total = num_params * param_bytes
    optimizer_total = num_params * optimizer_bytes_per_param

    result = {
        "gpu": 0,
        "cpu": 0,
        "nvme": 0,
    }

    if offload_param == "nvme":
        result["nvme"] += param_total
    else:
        result["gpu"] += param_total

    if offload_optimizer == "cpu":
        result["cpu"] += optimizer_total
    elif offload_optimizer == "nvme":
        result["nvme"] += optimizer_total
    else:
        result["gpu"] += optimizer_total

    return result
