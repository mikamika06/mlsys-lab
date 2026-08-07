def zero_plus_comm_volume(num_params, bytes_per_param, world_size, num_nodes, enable_hpz=True, enable_qgz=True):
    base_volume = 2.0 * num_params * bytes_per_param * (world_size - 1.0) / world_size
    hpz_factor = 0.5 if (enable_hpz and num_nodes > 1) else 1.0
    qgz_factor = 0.5 if enable_qgz else 1.0
    return base_volume * hpz_factor * qgz_factor
