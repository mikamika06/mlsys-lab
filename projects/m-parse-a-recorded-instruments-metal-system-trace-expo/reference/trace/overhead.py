"""Feed-dict overhead vs cached-buffer reuse comparison."""


def analyze_feed_dict_overhead(num_iterations, num_tensors, tensor_bytes, copy_bandwidth_gbps):
    bytes_per_iter = num_tensors * tensor_bytes
    total_bytes = bytes_per_iter * num_iterations
    bandwidth_bytes_per_us = (copy_bandwidth_gbps * 1e9) / 1e6
    time_per_iter_us = (bytes_per_iter / bandwidth_bytes_per_us) + 5.0
    total_time_us = time_per_iter_us * num_iterations
    return {
        "total_bytes_copied": total_bytes,
        "total_time_us": total_time_us,
        "per_iter_time_us": time_per_iter_us,
    }


def analyze_cached_reuse(num_iterations, num_tensors, tensor_bytes):
    time_per_iter_us = 1.0
    total_time_us = time_per_iter_us * num_iterations
    return {
        "total_bytes_copied": 0,
        "total_time_us": total_time_us,
        "per_iter_time_us": time_per_iter_us,
    }


def compute_reuse_speedup(num_iterations, num_tensors, tensor_bytes, copy_bandwidth_gbps):
    feed = analyze_feed_dict_overhead(num_iterations, num_tensors, tensor_bytes, copy_bandwidth_gbps)
    cached = analyze_cached_reuse(num_iterations, num_tensors, tensor_bytes)
    if cached["total_time_us"] == 0:
        return 1.0
    return feed["total_time_us"] / cached["total_time_us"]
