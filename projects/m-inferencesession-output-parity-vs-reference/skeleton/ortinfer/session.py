def run_inference(model_bytes, inputs, intra_op_num_threads=1, opt_level="BASIC"):
    raise NotImplementedError


def get_optimized_node_count(model_bytes, opt_level="BASIC"):
    raise NotImplementedError


def measure_latency_scaling(model_bytes, inputs, thread_counts):
    raise NotImplementedError
