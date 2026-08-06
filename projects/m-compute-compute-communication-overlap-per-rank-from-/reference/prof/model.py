def compute_comm_bound_ratio(timings, model_params):
    step_time = float(timings["step_time"])
    if step_time <= 0.0:
        return 0.0
    msg_size = float(model_params["msg_size_bytes"])
    world_size = float(model_params["world_size"])
    bandwidth = float(model_params["bandwidth_bytes_per_sec"])

    comm_time = 2.0 * ((world_size - 1.0) / world_size) * (msg_size / bandwidth)
    return (comm_time / step_time) * 100.0
