def compute_latency_ratio(model_name, torch_latencies, tvm_latencies):
    import numpy as np
    t_torch = float(np.mean(torch_latencies))
    t_tvm = float(np.mean(tvm_latencies))
    if t_tvm <= 0:
        return 0.0
    return t_torch / t_tvm
