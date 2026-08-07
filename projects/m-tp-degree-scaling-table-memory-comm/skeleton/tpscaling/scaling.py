def compute_tp_scaling_table(model_config, tp_degrees, sequence_length, batch_size, precision_bytes=2):
    raise NotImplementedError


def find_optimal_tp_degree(model_config, available_tp_degrees, sequence_length, batch_size, precision_bytes, gpu_memory_bytes, interconnect_bandwidth_bytes_per_sec, compute_flops_per_sec):
    raise NotImplementedError
