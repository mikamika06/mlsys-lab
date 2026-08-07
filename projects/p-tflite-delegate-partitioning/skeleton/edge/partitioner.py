def find_unsupported_ops(model_path):
    raise NotImplementedError

def rewrite_graph(model_path, out_path):
    raise NotImplementedError

def measure_latency_ratio(model_path, optimized_path):
    raise NotImplementedError

def verify_output_parity(model_path, optimized_path):
    raise NotImplementedError
