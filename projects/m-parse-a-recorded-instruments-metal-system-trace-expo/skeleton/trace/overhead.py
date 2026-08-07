"""Feed-dict overhead vs cached-buffer reuse comparison."""


def analyze_feed_dict_overhead(num_iterations, num_tensors, tensor_bytes, copy_bandwidth_gbps):
    raise NotImplementedError


def analyze_cached_reuse(num_iterations, num_tensors, tensor_bytes):
    raise NotImplementedError


def compute_reuse_speedup(num_iterations, num_tensors, tensor_bytes, copy_bandwidth_gbps):
    raise NotImplementedError
